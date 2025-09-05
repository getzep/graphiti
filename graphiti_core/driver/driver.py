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
from enum import Enum
from typing import Any

from opensearchpy import OpenSearch, helpers

logger = logging.getLogger(__name__)

DEFAULT_SIZE = 10


class GraphProvider(Enum):
    NEO4J = 'neo4j'
    FALKORDB = 'falkordb'
    KUZU = 'kuzu'
    NEPTUNE = 'neptune'


aoss_indices = [
    {
        'index_name': 'node_name_and_summary',
        'body': {
            'mappings': {
                'properties': {
                    'uuid': {'type': 'keyword'},
                    'name': {'type': 'text'},
                    'summary': {'type': 'text'},
                    'group_id': {'type': 'text'},
                    'created_at': {'type': 'date', 'format': "yyyy-MM-dd'T'HH:mm:ss.SSSZ"},
                    'name_embedding': {
                        'type': 'dense_vector',
                        'dims': 1024,
                        'index': True,
                        'similarity': 'cosine',
                    },
                }
            }
        },
        'query': {
            'query': {'multi_match': {'query': '', 'fields': ['name', 'summary', 'group_id']}},
            'size': DEFAULT_SIZE,
            'knn': {
                'field': 'name_embedding',
                'query_vector': [],
                'k': DEFAULT_SIZE,
                'num_candidates': 100,
            },
        },
    },
    {
        'index_name': 'community_name',
        'body': {
            'mappings': {
                'properties': {
                    'uuid': {'type': 'keyword'},
                    'name': {'type': 'text'},
                    'group_id': {'type': 'text'},
                }
            }
        },
        'query': {
            'query': {'multi_match': {'query': '', 'fields': ['name', 'group_id']}},
            'size': DEFAULT_SIZE,
        },
    },
    {
        'index_name': 'episode_content',
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
        'query': {
            'query': {
                'multi_match': {
                    'query': '',
                    'fields': ['content', 'source', 'source_description', 'group_id'],
                }
            },
            'size': DEFAULT_SIZE,
        },
    },
    {
        'index_name': 'edge_name_and_fact',
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
                        'type': 'dense_vector',
                        'dims': 1024,
                        'index': True,
                        'similarity': 'cosine',
                    },
                }
            }
        },
        'query': {
            'query': {'multi_match': {'query': '', 'fields': ['name', 'fact', 'group_id']}},
            'size': DEFAULT_SIZE,
            'knn': {
                'field': 'fact_embedding',
                'query_vector': [],  # supply vector at runtime
                'k': DEFAULT_SIZE,
                'num_candidates': 100,
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
    aoss_client: OpenSearch | None

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
        for index in aoss_indices:
            index_name = index['index_name']
            client = self.aoss_client
            if not client.indices.exists(index=index_name):
                client.indices.create(index=index_name, body=index['body'])
        # Sleep for 1 minute to let the index creation complete
        await asyncio.sleep(60)

    async def delete_aoss_indices(self):
        for index in aoss_indices:
            index_name = index['index_name']
            client = self.aoss_client
            if client.indices.exists(index=index_name):
                client.indices.delete(index=index_name)

    def run_aoss_query(self, name: str, query_text: str, limit: int = 10) -> dict[str, Any]:
        for index in aoss_indices:
            if name.lower() == index['index_name']:
                index['query']['query']['multi_match']['query'] = query_text
                query = {'size': limit, 'query': index['query']}
                resp = self.aoss_client.search(body=query['query'], index=index['index_name'])
                return resp
        return {}

    def save_to_aoss(self, name: str, data: list[dict]) -> int:
        for index in aoss_indices:
            if name.lower() == index['index_name']:
                to_index = []
                for d in data:
                    item = {'_index': name}
                    for p in index['body']['mappings']['properties']:
                        item[p] = d[p]
                    to_index.append(item)
                success, failed = helpers.bulk(self.aoss_client, to_index, stats_only=True)
                if failed > 0:
                    return success
                else:
                    return 0

        return 0
