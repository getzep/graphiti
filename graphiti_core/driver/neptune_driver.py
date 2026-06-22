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
from opensearchpy import OpenSearch, Urllib3AWSV4SignerAuth, Urllib3HttpConnection, helpers

from graphiti_core.driver.driver import GraphDriver, GraphDriverSession, GraphProvider
from graphiti_core.driver.neptune.operations.community_edge_ops import (
    NeptuneCommunityEdgeOperations,
)
from graphiti_core.driver.neptune.operations.community_node_ops import (
    NeptuneCommunityNodeOperations,
)
from graphiti_core.driver.neptune.operations.entity_edge_ops import NeptuneEntityEdgeOperations
from graphiti_core.driver.neptune.operations.entity_node_ops import NeptuneEntityNodeOperations
from graphiti_core.driver.neptune.operations.episode_node_ops import NeptuneEpisodeNodeOperations
from graphiti_core.driver.neptune.operations.episodic_edge_ops import NeptuneEpisodicEdgeOperations
from graphiti_core.driver.neptune.operations.graph_ops import NeptuneGraphMaintenanceOperations
from graphiti_core.driver.neptune.operations.has_episode_edge_ops import (
    NeptuneHasEpisodeEdgeOperations,
)
from graphiti_core.driver.neptune.operations.next_episode_edge_ops import (
    NeptuneNextEpisodeEdgeOperations,
)
from graphiti_core.driver.neptune.operations.saga_node_ops import NeptuneSagaNodeOperations
from graphiti_core.driver.neptune.operations.search_ops import NeptuneSearchOperations
from graphiti_core.driver.operations.community_edge_ops import CommunityEdgeOperations
from graphiti_core.driver.operations.community_node_ops import CommunityNodeOperations
from graphiti_core.driver.operations.entity_edge_ops import EntityEdgeOperations
from graphiti_core.driver.operations.entity_node_ops import EntityNodeOperations
from graphiti_core.driver.operations.episode_node_ops import EpisodeNodeOperations
from graphiti_core.driver.operations.episodic_edge_ops import EpisodicEdgeOperations
from graphiti_core.driver.operations.graph_ops import GraphMaintenanceOperations
from graphiti_core.driver.operations.has_episode_edge_ops import HasEpisodeEdgeOperations
from graphiti_core.driver.operations.next_episode_edge_ops import NextEpisodeEdgeOperations
from graphiti_core.driver.operations.saga_node_ops import SagaNodeOperations
from graphiti_core.driver.operations.search_ops import SearchOperations

logger = logging.getLogger(__name__)
DEFAULT_SIZE = 10

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

        # Instantiate Neptune operations
        self._entity_node_ops = NeptuneEntityNodeOperations()
        self._episode_node_ops = NeptuneEpisodeNodeOperations()
        self._community_node_ops = NeptuneCommunityNodeOperations(driver=self)
        self._saga_node_ops = NeptuneSagaNodeOperations()
        self._entity_edge_ops = NeptuneEntityEdgeOperations()
        self._episodic_edge_ops = NeptuneEpisodicEdgeOperations()
        self._community_edge_ops = NeptuneCommunityEdgeOperations()
        self._has_episode_edge_ops = NeptuneHasEpisodeEdgeOperations()
        self._next_episode_edge_ops = NeptuneNextEpisodeEdgeOperations()
        self._search_ops = NeptuneSearchOperations(driver=self)
        self._graph_ops = NeptuneGraphMaintenanceOperations(driver=self)

    # --- Operations properties ---

    @property
    def entity_node_ops(self) -> EntityNodeOperations:
        return self._entity_node_ops

    @property
    def episode_node_ops(self) -> EpisodeNodeOperations:
        return self._episode_node_ops

    @property
    def community_node_ops(self) -> CommunityNodeOperations:
        return self._community_node_ops

    @property
    def saga_node_ops(self) -> SagaNodeOperations:
        return self._saga_node_ops

    @property
    def entity_edge_ops(self) -> EntityEdgeOperations:
        return self._entity_edge_ops

    @property
    def episodic_edge_ops(self) -> EpisodicEdgeOperations:
        return self._episodic_edge_ops

    @property
    def community_edge_ops(self) -> CommunityEdgeOperations:
        return self._community_edge_ops

    @property
    def has_episode_edge_ops(self) -> HasEpisodeEdgeOperations:
        return self._has_episode_edge_ops

    @property
    def next_episode_edge_ops(self) -> NextEpisodeEdgeOperations:
        return self._next_episode_edge_ops

    @property
    def search_ops(self) -> SearchOperations:
        return self._search_ops

    @property
    def graph_ops(self) -> GraphMaintenanceOperations:
        return self._graph_ops

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
    ) -> tuple[list[dict[str, Any]], None, None]:
        params = dict(kwargs)
        if isinstance(cypher_query_, list):
            result: list[dict[str, Any]] = []
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

    def delete_all_indexes(self) -> Coroutine[Any, Any, Any]:
        return self.delete_all_indexes_impl()

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

    async def build_indices_and_constraints(self, delete_existing: bool = False):
        # Neptune uses OpenSearch (AOSS) for indexing
        if delete_existing:
            await self.delete_aoss_indices()
        await self.create_aoss_indices()

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
                    item = {'_index': name, '_id': d['uuid']}
                    for p in index['body']['mappings']['properties']:
                        if p in d:
                            item[p] = d[p]
                    to_index.append(item)
                success, failed = helpers.bulk(self.aoss_client, to_index, stats_only=True)
                return success

        return 0


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
