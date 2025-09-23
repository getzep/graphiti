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
import os
from abc import ABC, abstractmethod
from collections.abc import Coroutine
from datetime import datetime
from enum import Enum
from typing import Any

from dotenv import load_dotenv

from graphiti_core.embedder.client import EMBEDDING_DIM

try:
    from opensearchpy import AsyncOpenSearch, helpers

    _HAS_OPENSEARCH = True
except ImportError:
    OpenSearch = None
    helpers = None
    _HAS_OPENSEARCH = False

logger = logging.getLogger(__name__)

DEFAULT_SIZE = 10

load_dotenv()

ENTITY_INDEX_NAME = os.environ.get('ENTITY_INDEX_NAME', 'entities')
EPISODE_INDEX_NAME = os.environ.get('EPISODE_INDEX_NAME', 'episodes')
COMMUNITY_INDEX_NAME = os.environ.get('COMMUNITY_INDEX_NAME', 'communities')
ENTITY_EDGE_INDEX_NAME = os.environ.get('ENTITY_EDGE_INDEX_NAME', 'entity_edges')


class GraphProvider(Enum):
    NEO4J = 'neo4j'
    FALKORDB = 'falkordb'
    KUZU = 'kuzu'
    NEPTUNE = 'neptune'


aoss_indices = [
    {
        'index_name': ENTITY_INDEX_NAME,
        'body': {
            'settings': {'index': {'knn': True}},
            'mappings': {
                'properties': {
                    'uuid': {'type': 'keyword'},
                    'name': {'type': 'text'},
                    'summary': {'type': 'text'},
                    'group_id': {'type': 'keyword'},
                    'created_at': {'type': 'date', 'format': 'strict_date_optional_time_nanos'},
                    'name_embedding': {
                        'type': 'knn_vector',
                        'dimension': EMBEDDING_DIM,
                        'method': {
                            'engine': 'faiss',
                            'space_type': 'cosinesimil',
                            'name': 'hnsw',
                            'parameters': {'ef_construction': 128, 'm': 16},
                        },
                    },
                }
            },
        },
    },
    {
        'index_name': COMMUNITY_INDEX_NAME,
        'body': {
            'mappings': {
                'properties': {
                    'uuid': {'type': 'keyword'},
                    'name': {'type': 'text'},
                    'group_id': {'type': 'keyword'},
                }
            }
        },
    },
    {
        'index_name': EPISODE_INDEX_NAME,
        'body': {
            'mappings': {
                'properties': {
                    'uuid': {'type': 'keyword'},
                    'content': {'type': 'text'},
                    'source': {'type': 'text'},
                    'source_description': {'type': 'text'},
                    'group_id': {'type': 'keyword'},
                    'created_at': {'type': 'date', 'format': 'strict_date_optional_time_nanos'},
                    'valid_at': {'type': 'date', 'format': 'strict_date_optional_time_nanos'},
                }
            }
        },
    },
    {
        'index_name': ENTITY_EDGE_INDEX_NAME,
        'body': {
            'settings': {'index': {'knn': True}},
            'mappings': {
                'properties': {
                    'uuid': {'type': 'keyword'},
                    'name': {'type': 'text'},
                    'fact': {'type': 'text'},
                    'group_id': {'type': 'keyword'},
                    'created_at': {'type': 'date', 'format': 'strict_date_optional_time_nanos'},
                    'valid_at': {'type': 'date', 'format': 'strict_date_optional_time_nanos'},
                    'expired_at': {'type': 'date', 'format': 'strict_date_optional_time_nanos'},
                    'invalid_at': {'type': 'date', 'format': 'strict_date_optional_time_nanos'},
                    'fact_embedding': {
                        'type': 'knn_vector',
                        'dimension': EMBEDDING_DIM,
                        'method': {
                            'engine': 'faiss',
                            'space_type': 'cosinesimil',
                            'name': 'hnsw',
                            'parameters': {'ef_construction': 128, 'm': 16},
                        },
                    },
                }
            },
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
    aoss_client: AsyncOpenSearch | None  # type: ignore

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
            if await client.indices.exists_alias(name=alias_name):
                continue

            # Build a physical index name with timestamp
            ts_suffix = datetime.utcnow().strftime('%Y%m%d%H%M%S')
            physical_index_name = f'{alias_name}_{ts_suffix}'

            # Create the index
            await client.indices.create(index=physical_index_name, body=index['body'])

            # Point alias to it
            await client.indices.put_alias(index=physical_index_name, name=alias_name)

        # Allow some time for index creation
        await asyncio.sleep(1)

    async def delete_aoss_indices(self):
        client = self.aoss_client

        if not client:
            logger.warning('No OpenSearch client found')
            return

        for entry in aoss_indices:
            alias_name = entry['index_name']

            try:
                # Resolve alias → indices
                alias_info = await client.indices.get_alias(name=alias_name)
                indices = list(alias_info.keys())

                if not indices:
                    logger.info(f"No indices found for alias '{alias_name}'")
                    continue

                for index in indices:
                    if await client.indices.exists(index=index):
                        await client.indices.delete(index=index)
                        logger.info(f"Deleted index '{index}' (alias: {alias_name})")
                    else:
                        logger.warning(f"Index '{index}' not found for alias '{alias_name}'")

            except Exception as e:
                logger.error(f"Error deleting indices for alias '{alias_name}': {e}")

    async def clear_aoss_indices(self):
        client = self.aoss_client

        if not client:
            logger.warning('No OpenSearch client found')
            return

        for index in aoss_indices:
            index_name = index['index_name']

            if await client.indices.exists(index=index_name):
                try:
                    # Delete all documents but keep the index
                    response = await client.delete_by_query(
                        index=index_name,
                        body={'query': {'match_all': {}}},
                    )
                    logger.info(f"Cleared index '{index_name}': {response}")
                except Exception as e:
                    logger.error(f"Error clearing index '{index_name}': {e}")
            else:
                logger.warning(f"Index '{index_name}' does not exist")

    async def save_to_aoss(self, name: str, data: list[dict]) -> int:
        client = self.aoss_client
        if not client or not helpers:
            logger.warning('No OpenSearch client found')
            return 0

        for index in aoss_indices:
            if name.lower() == index['index_name']:
                to_index = []
                for d in data:
                    doc = {}
                    for p in index['body']['mappings']['properties']:
                        if p in d:  # protect against missing fields
                            doc[p] = d[p]

                    item = {
                        '_index': name,
                        '_id': d['uuid'],
                        '_routing': d.get('group_id'),
                        '_source': doc,
                    }
                    to_index.append(item)

                success, failed = await helpers.async_bulk(
                    client, to_index, stats_only=True, request_timeout=60
                )

                return success if failed == 0 else success

        return 0

    def build_fulltext_query(
        self, query: str, group_ids: list[str] | None = None, max_query_length: int = 128
    ) -> str:
        """
        Specific fulltext query builder for database providers.
        Only implemented by providers that need custom fulltext query building.
        """
        raise NotImplementedError(f'build_fulltext_query not implemented for {self.provider}')
