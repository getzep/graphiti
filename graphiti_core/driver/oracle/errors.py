# Copyright 2026, Zep Software, Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from __future__ import annotations

from typing import Any


class Oracle26aiError(RuntimeError):
    """Base error for the Oracle Graphiti adapter."""


class OracleQueryContractError(Oracle26aiError):
    """Raised when a query does not use the driver's Oracle SQL contract."""


class OracleConnectionError(Oracle26aiError):
    """Raised when an Oracle dependency cannot be reached safely."""


class InvalidEmbeddingDimension(Oracle26aiError):
    def __init__(self, *, expected: int, actual: int) -> None:
        super().__init__(f'invalid embedding dimension: expected {expected}, received {actual}')
        self.expected = expected
        self.actual = actual


class NaiveDatetimeError(Oracle26aiError):
    """Raised when a timestamp has no timezone information."""


class InvalidEdgeEndpoint(Oracle26aiError):
    """Raised when an edge points to absent or incompatible node kinds."""


class GroupScopeConflict(Oracle26aiError):
    """Raised when a UUID already belongs to another graph group."""


class EpisodeRemovalConflict(Oracle26aiError):
    """Retry removal after source changes; repair missing provenance before retrying."""


class EpisodeSummaryRebuildError(Oracle26aiError):
    """Summary preparation failed before the episode removal could mutate persistence."""


class InvalidSearchFilter(Oracle26aiError):
    """Raised when a search filter cannot be represented safely in Oracle SQL."""


class InvalidSearchQuery(Oracle26aiError):
    """Raised when a text search has no safe searchable terms."""


class InvalidTraversalDepth(Oracle26aiError):
    """Raised when a SQL/PGQ traversal is not bounded to the supported depth."""


def oracle_error_code(error: BaseException) -> int | None:
    """Return a numeric Oracle code without serializing the exception payload."""

    if not getattr(error, 'args', None):
        return None
    detail: Any = error.args[0]
    code = getattr(detail, 'code', None)
    return int(code) if isinstance(code, int) else None
