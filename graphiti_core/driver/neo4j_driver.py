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

from neo4j import AsyncGraphDatabase, EagerResult
from typing_extensions import LiteralString

from graphiti_core.driver.driver import GraphDriver, GraphDriverSession, GraphProvider

logger = logging.getLogger(__name__)


class Neo4jDriver(GraphDriver):
    provider = GraphProvider.NEO4J

    def __init__(self, uri: str, user: str | None, password: str | None, database: str = 'neo4j'):
        super().__init__()
        self.client = AsyncGraphDatabase.driver(
            uri=uri,
            auth=(user or '', password or ''),
        )
        self._database = database

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
        return self.client.execute_query(
            'CALL db.indexes() YIELD name DROP INDEX name',
        )
    
    def sanitize(self, query: str) -> str:
        # Escape special characters from a query before passing into Lucene
        # + - && || ! ( ) { } [ ] ^ " ~ * ? : \ /
        escape_map = str.maketrans(
            {
                '+': r'\+',
                '-': r'\-',
                '&': r'\&',
                '|': r'\|',
                '!': r'\!',
                '(': r'\(',
                ')': r'\)',
                '{': r'\{',
                '}': r'\}',
                '[': r'\[',
                ']': r'\]',
                '^': r'\^',
                '"': r'\"',
                '~': r'\~',
                '*': r'\*',
                '?': r'\?',
                ':': r'\:',
                '\\': r'\\',
                '/': r'\/',
                'O': r'\O',
                'R': r'\R',
                'N': r'\N',
                'T': r'\T',
                'A': r'\A',
                'D': r'\D',
            }
        )

        sanitized = query.translate(escape_map)
        return sanitized

    def build_fulltext_query(self, query: str, group_ids: list[str] | None = None, max_query_length: int = 128) -> str:
        """
        Build a fulltext query string for Neo4j.
        Neo4j uses Lucene syntax where string values need to be wrapped in single quotes.
        """
        # Lucene expects string values (e.g. group_id) to be wrapped in single quotes
        group_ids_filter_list = (
            [self.fulltext_syntax + f'group_id:"{g}"' for g in group_ids] if group_ids is not None else []
        )
        group_ids_filter = ''
        for f in group_ids_filter_list:
            group_ids_filter += f if not group_ids_filter else f' OR {f}'

        group_ids_filter += ' AND ' if group_ids_filter else ''

        lucene_query = self.sanitize(query)
        # If the lucene query is too long return no query
        if len(lucene_query.split(' ')) + len(group_ids or '') >= max_query_length:
            return ''

        full_query = group_ids_filter + '(' + lucene_query + ')'

        return full_query
