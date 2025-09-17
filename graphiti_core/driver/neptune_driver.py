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
import datetime
import logging
from collections.abc import Coroutine
from typing import Any

import boto3
from langchain_aws.graphs import NeptuneAnalyticsGraph, NeptuneGraph
from opensearchpy import OpenSearch, Urllib3AWSV4SignerAuth, Urllib3HttpConnection

from graphiti_core.driver.driver import (
    DEFAULT_SIZE,
    GraphDriver,
    GraphDriverSession,
    GraphProvider,
)

logger = logging.getLogger(__name__)

neptune_aoss_indices = [
    {
        'index_name': 'node_name_and_summary',
        'alias_name': 'entities',
        'body': {
            'mappings': {
                'properties': {
                    'uuid': {'type': 'keyword'},
                    'name': {'type': 'text'},
                    'summary': {'type': 'text'},
                    'group_id': {'type': 'text'},
                }
            }
        },
        'query': {
            'query': {'multi_match': {'query': '', 'fields': ['name', 'summary', 'group_id']}},
            'size': DEFAULT_SIZE,
        },
    },
    {
        'index_name': 'community_name',
        'alias_name': 'communities',
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
        'alias_name': 'episodes',
        'body': {
            'mappings': {
                'properties': {
                    'uuid': {'type': 'keyword'},
                    'content': {'type': 'text'},
                    'source': {'type': 'text'},
                    'source_description': {'type': 'text'},
                    'group_id': {'type': 'text'},
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
        'alias_name': 'facts',
        'body': {
            'mappings': {
                'properties': {
                    'uuid': {'type': 'keyword'},
                    'name': {'type': 'text'},
                    'fact': {'type': 'text'},
                    'group_id': {'type': 'text'},
                }
            }
        },
        'query': {
            'query': {'multi_match': {'query': '', 'fields': ['name', 'fact', 'group_id']}},
            'size': DEFAULT_SIZE,
        },
    },
]


class NeptuneDriver(GraphDriver):
    provider: GraphProvider = GraphProvider.NEPTUNE

    def __init__(self, host: str, aoss_host: str, port: int = 8182, aoss_port: int = 443):
        """This initializes a NeptuneDriver for use with Neptune as a backend

        Args:
            host (str): The Neptune Database or Neptune Analytics host
            aoss_host (str): The OpenSearch host value
            port (int, optional): The Neptune Database port, ignored for Neptune Analytics. Defaults to 8182.
            aoss_port (int, optional): The OpenSearch port. Defaults to 443.
        """
        if not host:
            raise ValueError('You must provide an endpoint to create a NeptuneDriver')

        if host.startswith('neptune-db://'):
            # This is a Neptune Database Cluster
            endpoint = host.replace('neptune-db://', '')
            self.client = NeptuneGraph(endpoint, port)
            logger.debug('Creating Neptune Database session for %s', host)
        elif host.startswith('neptune-graph://'):
            # This is a Neptune Analytics Graph
            graphId = host.replace('neptune-graph://', '')
            self.client = NeptuneAnalyticsGraph(graphId)
            logger.debug('Creating Neptune Graph session for %s', host)
        else:
            raise ValueError(
                'You must provide an endpoint to create a NeptuneDriver as either neptune-db://<endpoint> or neptune-graph://<graphid>'
            )

        if not aoss_host:
            raise ValueError('You must provide an AOSS endpoint to create an OpenSearch driver.')

        session = boto3.Session()
        self.aoss_client = OpenSearch(
            hosts=[{'host': aoss_host, 'port': aoss_port}],
            http_auth=Urllib3AWSV4SignerAuth(
                session.get_credentials(), session.region_name, 'aoss'
            ),
            use_ssl=True,
            verify_certs=True,
            connection_class=Urllib3HttpConnection,
            pool_maxsize=20,
        )

    def _sanitize_parameters(self, query, params: dict):
        if isinstance(query, list):
            queries = []
            for q in query:
                queries.append(self._sanitize_parameters(q, params))
            return queries
        else:
            for k, v in params.items():
                if isinstance(v, datetime.datetime):
                    params[k] = v.isoformat()
                elif isinstance(v, list):
                    # Handle lists that might contain datetime objects
                    for i, item in enumerate(v):
                        if isinstance(item, datetime.datetime):
                            v[i] = item.isoformat()
                            query = str(query).replace(f'${k}', f'datetime(${k})')
                        if isinstance(item, dict):
                            query = self._sanitize_parameters(query, v[i])

                    # If the list contains datetime objects, we need to wrap each element with datetime()
                    if any(isinstance(item, str) and 'T' in item for item in v):
                        # Create a new list expression with datetime() wrapped around each element
                        datetime_list = (
                            '['
                            + ', '.join(
                                f'datetime("{item}")'
                                if isinstance(item, str) and 'T' in item
                                else repr(item)
                                for item in v
                            )
                            + ']'
                        )
                        query = str(query).replace(f'${k}', datetime_list)
                elif isinstance(v, dict):
                    query = self._sanitize_parameters(query, v)
            return query

    async def execute_query(
        self, cypher_query_, **kwargs: Any
    ) -> tuple[dict[str, Any], None, None]:
        params = dict(kwargs)
        if isinstance(cypher_query_, list):
            for q in cypher_query_:
                result, _, _ = self._run_query(q[0], q[1])
            return result, None, None
        else:
            return self._run_query(cypher_query_, params)

    def _run_query(self, cypher_query_, params):
        cypher_query_ = str(self._sanitize_parameters(cypher_query_, params))
        try:
            result = self.client.query(cypher_query_, params=params)
        except Exception as e:
            logger.error('Query: %s', cypher_query_)
            logger.error('Parameters: %s', params)
            logger.error('Error executing query: %s', e)
            raise e

        return result, None, None

    def session(self, database: str | None = None) -> GraphDriverSession:
        return NeptuneDriverSession(driver=self)

    async def close(self) -> None:
        return self.client.client.close()

    async def _delete_all_data(self) -> Any:
        return await self.execute_query('MATCH (n) DETACH DELETE n')

    async def create_aoss_indices(self):
        for index in neptune_aoss_indices:
            index_name = index['index_name']
            client = self.aoss_client
            if not client:
                raise ValueError(
                    'You must provide an AOSS endpoint to create an OpenSearch driver.'
                )
            if not client.indices.exists(index=index_name):
                await client.indices.create(index=index_name, body=index['body'])

            alias_name = index.get('alias_name', index_name)

            if not client.indices.exists_alias(name=alias_name, index=index_name):
                await client.indices.put_alias(index=index_name, name=alias_name)

        # Sleep for 1 minute to let the index creation complete
        await asyncio.sleep(60)

    def delete_all_indexes(self) -> Coroutine[Any, Any, Any]:
        return self.delete_all_indexes_impl()


class NeptuneDriverSession(GraphDriverSession):
    provider = GraphProvider.NEPTUNE

    def __init__(self, driver: NeptuneDriver):  # type: ignore[reportUnknownArgumentType]
        self.driver = driver

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc, tb):
        # No cleanup needed for Neptune, but method must exist
        pass

    async def close(self):
        # No explicit close needed for Neptune, but method must exist
        pass

    async def execute_write(self, func, *args, **kwargs):
        # Directly await the provided async function with `self` as the transaction/session
        return await func(self, *args, **kwargs)

    async def run(self, query: str | list, **kwargs: Any) -> Any:
        if isinstance(query, list):
            res = None
            for q in query:
                res = await self.driver.execute_query(q, **kwargs)
            return res
        else:
            return await self.driver.execute_query(str(query), **kwargs)
