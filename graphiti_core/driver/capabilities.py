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

from pydantic import BaseModel


class GraphCapabilities(BaseModel):
    """What a graph backend supports natively.

    Connectors declare their capabilities so the search/index layer can branch on
    a *capability* rather than on backend identity (``provider == GraphProvider.X``).
    A backend that lacks a native capability can then be routed through a fallback
    path (external vector-store overlay, in-library BM25) instead of emitting
    backend-specific procedures it does not implement.

    Defaults are conservative — a bare capability set claims nothing native, which
    is the safe assumption for a minimal Bolt/Cypher backend.
    """

    supports_transactions: bool = False
    """Real commit/rollback semantics (as opposed to immediate-mode / autocommit)."""

    supports_native_fulltext_search: bool = False
    """Native fulltext search wired through the connector's search operations
    (e.g. Neo4j ``db.index.fulltext.*``)."""

    supports_native_vector_search: bool = False
    """Native vector similarity search wired through the connector's search
    operations."""

    supports_vector_index: bool = False
    """Can create/manage a persistent vector index via the query language."""
