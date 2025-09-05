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

import logging
from collections.abc import Coroutine
from typing import Any

import boto3
from neo4j import AsyncGraphDatabase, EagerResult
from opensearchpy import OpenSearch, Urllib3AWSV4SignerAuth, Urllib3HttpConnection
from typing_extensions import LiteralString

from graphiti_core.driver.driver import GraphDriver, GraphDriverSession, GraphProvider

logger = logging.getLogger(__name__)


class Neo4jDriver(GraphDriver):
    provider = GraphProvider.NEO4J

    def __init__(
        self,
        uri: str,
        user: str | None,
        password: str | None,
        database: str = 'neo4j',
        aoss_host: str | None = None,
        aoss_port: int | None = None,
    ):
        super().__init__()
        self.client = AsyncGraphDatabase.driver(
            uri=uri,
            auth=(user or '', password or ''),
        )
        self._database = database

        self.aoss_client = None
        if aoss_host and aoss_port:
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

    async def execute_query(self, cypher_query_: LiteralString, **kwargs: Any) -> EagerResult:
        # Check if database_ is provided in kwargs.
        # If not populated, set the value to retain backwards compatibility
        params = kwargs.pop('params', None)
        if params is None:
            params = {}
        params.setdefault('database_', self._database)

        try:
            result = await self.client.execute_query(cypher_query_, parameters_=params, **kwargs)
        except Exception as e:
            logger.error(f'Error executing Neo4j query: {e}\n{cypher_query_}\n{params}')
            raise

        return result

    def session(self, database: str | None = None) -> GraphDriverSession:
        _database = database or self._database
        return self.client.session(database=_database)  # type: ignore

    async def close(self) -> None:
        return await self.client.close()

    def delete_all_indexes(self) -> Coroutine[Any, Any, EagerResult]:
        if self.aoss_client:
            self.delete_all_indexes_impl()
        return self.client.execute_query(
            'CALL db.indexes() YIELD name DROP INDEX name',
        )
