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

from collections.abc import Iterator, Sequence
from typing import TYPE_CHECKING, TypeVar

from graphiti_core.driver.query_executor import QueryExecutor, Transaction
from graphiti_core.driver.spanner.errors import SpannerDriverError
from graphiti_core.driver.spanner.transaction import SpannerTransaction

if TYPE_CHECKING:
    from graphiti_core.driver.spanner.driver import SpannerDriver

T = TypeVar('T')

# IN UNNEST parameter lists are chunked so a single read stays small.
READ_CHUNK = 500


def chunked(items: Sequence[T], size: int) -> Iterator[Sequence[T]]:
    """Yield ``items`` in consecutive slices of at most ``size`` elements."""
    size = max(size, 1)
    for start in range(0, len(items), size):
        yield items[start : start + size]


def spanner_driver(executor: QueryExecutor) -> SpannerDriver:
    """Return ``executor`` typed as a :class:`SpannerDriver`.

    The ops ABCs receive a generic ``QueryExecutor``; Spanner ops need the driver's
    ``read``/``apply_mutations``/``execute_partitioned_dml`` API. Raises
    :class:`SpannerDriverError` for any other executor.
    """
    # Imported here: driver.py imports the ops modules, which import this module.
    from graphiti_core.driver.spanner.driver import SpannerDriver

    if not isinstance(executor, SpannerDriver):
        raise SpannerDriverError(
            f'Spanner operations require a SpannerDriver executor, got {type(executor).__name__}'
        )
    return executor


def spanner_transaction(tx: Transaction | None) -> SpannerTransaction | None:
    """Return ``tx`` typed as a :class:`SpannerTransaction` (or ``None``).

    Raises :class:`SpannerDriverError` for a transaction from another driver.
    """
    if tx is None or isinstance(tx, SpannerTransaction):
        return tx
    raise SpannerDriverError(
        f'Spanner operations require a SpannerTransaction, got {type(tx).__name__}'
    )
