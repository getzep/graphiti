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

from collections.abc import Awaitable, Callable
from typing import TYPE_CHECKING, Any

from pydantic import BaseModel, ConfigDict, Field, model_validator

if TYPE_CHECKING:
    from graphiti_core.ingestion.state_store import IngestionStateStore

# before_commit_write(change_set) -> change_set
BeforeCommitWriteHook = Callable[[Any], Awaitable[Any]]
# before_enrichment_write(nodes, unit) -> nodes
BeforeEnrichmentWriteHook = Callable[[list[Any], Any], Awaitable[list[Any]]]


class ConcurrentIngestionConfig(BaseModel):
    """Settings of concurrent ingestion for one graph.

    The mode is off when `Graphiti` gets no object of this class.

    Parameters
    ----------
    window : int
        The largest number of units of one graph that run at one time in this process.
        Each unit also runs its model calls through `semaphore_gather`, so the largest
        number of model calls at one time is `window` times `SEMAPHORE_LIMIT`.
    state_store : IngestionStateStore | None
        The store of the sequence, the change sets, and the enrichment order. The default
        is a new `InMemoryIngestionStateStore`.
    lease_seconds : float
        The time after which a unit with no heartbeat is expired.
    write_margin_seconds : float
        The time before the local deadline at which a graph write must be complete.
    poll_interval_seconds : float
        The wait time between two store calls when a unit waits for a different process.
    index_lag_seconds : float
        The longest time between a completed graph write and the time when a search
        returns the record.
    max_apply_attempts : int
        The number of attempts to write one final change set.
    enabled_for_group : Callable[[str], bool] | None
        When the function returns False for a group_id, that call uses serial ingestion.
    before_commit_write : async Callable[[ChangeSet], ChangeSet] | None
        Runs after the repair and before the final change set is stored.
    before_enrichment_write : async Callable[[list[EntityNode], AdmittedUnit], list[EntityNode]] | None
        Runs before each enrichment write.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    window: int = Field(ge=1)
    state_store: Any = None
    lease_seconds: float = Field(default=60.0, gt=0)
    write_margin_seconds: float = Field(default=5.0, ge=0)
    poll_interval_seconds: float = Field(default=0.25, gt=0)
    index_lag_seconds: float = Field(default=300.0, ge=0)
    max_apply_attempts: int = Field(default=5, ge=1)
    enabled_for_group: Callable[[str], bool] | None = None
    before_commit_write: BeforeCommitWriteHook | None = None
    before_enrichment_write: BeforeEnrichmentWriteHook | None = None

    @model_validator(mode='after')
    def _check_values(self) -> 'ConcurrentIngestionConfig':
        if self.write_margin_seconds >= self.lease_seconds / 3:
            raise ValueError('write_margin_seconds must be less than lease_seconds / 3')
        if self.state_store is not None:
            from graphiti_core.ingestion.state_store import IngestionStateStore

            if not isinstance(self.state_store, IngestionStateStore):
                raise ValueError('state_store must be an IngestionStateStore')
        return self

    def get_state_store(self) -> 'IngestionStateStore':
        """Return the configured store. Create the in-memory store on the first call."""
        if self.state_store is None:
            from graphiti_core.ingestion.state_store import InMemoryIngestionStateStore

            self.state_store = InMemoryIngestionStateStore()
        return self.state_store
