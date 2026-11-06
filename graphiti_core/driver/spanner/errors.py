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

from __future__ import annotations

QUERY_SNIPPET_LENGTH = 120


def query_snippet(query: str, length: int = QUERY_SNIPPET_LENGTH) -> str:
    """First ``length`` characters of a query, with runs of whitespace collapsed."""
    return ' '.join(query.split())[:length]


class SpannerDriverError(Exception):
    """Base class for every error raised by the Spanner driver."""


class SpannerCypherNotSupportedError(SpannerDriverError):
    """``execute_query`` received Cypher. Spanner runs GoogleSQL and GQL only.

    Raised loudly on purpose: a Cypher string reaching the Spanner driver means a core code
    path fell back to its inline Cypher because a Spanner operation is not implemented yet.
    """

    def __init__(self, query: str, reason: str | None = None):
        self.query = query
        self.reason = reason
        snippet = query_snippet(query)
        detail = f' ({reason})' if reason else ''
        super().__init__(
            f'SpannerDriver cannot run Cypher{detail}. Use GoogleSQL or GQL '
            f'(GRAPH <graph> MATCH ...) with @name parameters, or route the call through the '
            f'Spanner operations classes. Query: {snippet}'
        )


class SpannerMissingEndpointError(SpannerDriverError):
    """An edge write referenced a source node row that does not exist.

    Edge tables are interleaved in their source node table, so Spanner rejects the whole
    commit when the source node is missing (Neo4j's ``MATCH ... MERGE`` silently writes
    nothing instead).
    """

    def __init__(
        self,
        message: str,
        *,
        table: str | None = None,
        edge_uuid: str | None = None,
        node_uuid: str | None = None,
    ):
        self.table = table
        self.edge_uuid = edge_uuid
        self.node_uuid = node_uuid
        super().__init__(message)


class SpannerSchemaError(SpannerDriverError):
    """A schema (DDL) operation failed or the database schema is not what the driver expects."""
