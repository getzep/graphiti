"""
Copyright 2024, Zep Software, Inc.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

import asyncio
import copy
import logging
from abc import ABC, abstractmethod
from collections.abc import Coroutine
from datetime import datetime
from enum import Enum
from typing import Any

from graphiti_core.embedder.client import EMBEDDING_DIM

try:
    from opensearchpy import OpenSearch, helpers

    _HAS_OPENSEARCH = True
except ImportError:
    OpenSearch = None
    helpers = None
    _HAS_OPENSEARCH = False

logger = logging.getLogger(__name__)

DEFAULT_SIZE = 10


class GraphProvider(Enum):
    NEO4J = 'neo4j'
    FALKORDB = 'falkordb'
    KUZU = 'kuzu'
    NEPTUNE = 'neptune'


aoss_indices = [
    {
        'index_name': 'entities',
        'body': {
            'mappings': {
                'properties': {
                    'uuid': {'type': 'keyword'},
                    'name': {'type': 'text'},
                    'summary': {'type': 'text'},
                    'group_id': {'type': 'text'},
                    'created_at': {'type': 'date', 'format': "yyyy-MM-dd'T'HH:mm:ss.SSSZ"},
                    'name_embedding': {
                        'type': 'knn_vector',
                        'dims': EMBEDDING_DIM,
                        'index': True,
                        'similarity': 'cosine',
                        'method': {
                            'engine': 'faiss',
                            'space_type': 'cosinesimil',
                            'name': 'hnsw',
                            'parameters': {'ef_construction': 128, 'm': 16},
                        },
                    },
                }
            }
        },
    },
    {
        'index_name': 'communities',
        'body': {
            'mappings': {
                'properties': {
                    'uuid': {'type': 'keyword'},
                    'name': {'type': 'text'},
                    'group_id': {'type': 'text'},
                }
            }
        },
    },
    {
        'index_name': 'episodes',
        'body': {
            'mappings': {
                'properties': {
                    'uuid': {'type': 'keyword'},
                    'content': {'type': 'text'},
                    'source': {'type': 'text'},
                    'source_description': {'type': 'text'},
                    'group_id': {'type': 'text'},
                    'created_at': {'type': 'date', 'format': "yyyy-MM-dd'T'HH:mm:ss.SSSZ"},
                    'valid_at': {'type': 'date', 'format': "yyyy-MM-dd'T'HH:mm:ss.SSSZ"},
                }
            }
        },
    },
    {
        'index_name': 'entity_edges',
        'body': {
            'mappings': {
                'properties': {
                    'uuid': {'type': 'keyword'},
                    'name': {'type': 'text'},
                    'fact': {'type': 'text'},
                    'group_id': {'type': 'text'},
                    'created_at': {'type': 'date', 'format': "yyyy-MM-dd'T'HH:mm:ss.SSSZ"},
                    'valid_at': {'type': 'date', 'format': "yyyy-MM-dd'T'HH:mm:ss.SSSZ"},
                    'expired_at': {'type': 'date', 'format': "yyyy-MM-dd'T'HH:mm:ss.SSSZ"},
                    'invalid_at': {'type': 'date', 'format': "yyyy-MM-dd'T'HH:mm:ss.SSSZ"},
                    'fact_embedding': {
                        'type': 'knn_vector',
                        'dims': EMBEDDING_DIM,
                        'index': True,
                        'similarity': 'cosine',
                        'method': {
                            'engine': 'faiss',
                            'space_type': 'cosinesimil',
                            'name': 'hnsw',
                            'parameters': {'ef_construction': 128, 'm': 16},
                        },
                    },
                }
            }
        },
    },
]


class GraphDriverSession(ABC):
    provider: GraphProvider

    async def __aenter__(self):
        return self

    @abstractmethod
    async def __aexit__(self, exc_type, exc, tb):
        # No cleanup needed for Falkor, but method must exist
        pass

    @abstractmethod
    async def run(self, query: str, **kwargs: Any) -> Any:
        raise NotImplementedError()

    @abstractmethod
    async def close(self):
        raise NotImplementedError()

    @abstractmethod
    async def execute_write(self, func, *args, **kwargs):
        raise NotImplementedError()


class GraphDriver(ABC):
    provider: GraphProvider
    fulltext_syntax: str = (
        ''  # Neo4j (default) syntax does not require a prefix for fulltext queries
    )
    _database: str
    aoss_client: OpenSearch | None  # type: ignore

    @abstractmethod
    def execute_query(self, cypher_query_: str, **kwargs: Any) -> Coroutine:
        raise NotImplementedError()

    @abstractmethod
    def session(self, database: str | None = None) -> GraphDriverSession:
        raise NotImplementedError()

    @abstractmethod
    def close(self):
        raise NotImplementedError()

    @abstractmethod
    def delete_all_indexes(self) -> Coroutine:
        raise NotImplementedError()

    def with_database(self, database: str) -> 'GraphDriver':
        """
        Returns a shallow copy of this driver with a different default database.
        Reuses the same connection (e.g. FalkorDB, Neo4j).
        """
        cloned = copy.copy(self)
        cloned._database = database

        return cloned

    async def delete_all_indexes_impl(self) -> Coroutine[Any, Any, Any]:
        # No matter what happens above, always return True
        return self.delete_aoss_indices()

    async def create_aoss_indices(self):
        client = self.aoss_client
        if not client:
            logger.warning('No OpenSearch client found')
            return

        for index in aoss_indices:
            alias_name = index['index_name']

            # If alias already exists, skip (idempotent behavior)
            if client.indices.exists_alias(name=alias_name):
                continue

            # Build a physical index name with timestamp
            ts_suffix = datetime.utcnow().strftime('%Y%m%d%H%M%S')
            physical_index_name = f'{alias_name}_{ts_suffix}'

            # Create the index
            client.indices.create(index=physical_index_name, body=index['body'])

            # Point alias to it
            client.indices.put_alias(index=physical_index_name, name=alias_name)

        # Allow some time for index creation
        await asyncio.sleep(60)

    async def delete_aoss_indices(self):
        for index in aoss_indices:
            index_name = index['index_name']
            client = self.aoss_client

            if not client:
                logger.warning('No OpenSearch client found')
                return

            if client.indices.exists(index=index_name):
                client.indices.delete(index=index_name)

    def save_to_aoss(self, name: str, data: list[dict]) -> int:
        client = self.aoss_client
        if not client or not helpers:
            logger.warning('No OpenSearch client found')
            return 0

        for index in aoss_indices:
            if name.lower() == index['index_name']:
                to_index = []
                for d in data:
                    item = {
                        '_index': name,
                        '_routing': d.get('group_id'),  # shard routing
                    }
                    for p in index['body']['mappings']['properties']:
                        if p in d:  # protect against missing fields
                            item[p] = d[p]
                    to_index.append(item)

                success, failed = helpers.bulk(client, to_index, stats_only=True)

                return success if failed == 0 else success

        return 0
