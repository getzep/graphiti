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

# Concurrent ingestion for one graph: change sets, state stores, phases, and the scheduler.

from graphiti_core.errors import (
    ChangeSetRejected,
    StaleCommitError,
    UnitExpiredError,
    UnitInProgressError,
)

from .change_set import (
    AddEpisodesToEdge,
    ChangeSet,
    EndEdge,
    PatchNode,
    ResolutionTrace,
    SagaLink,
    SetEdgeAttributes,
)
from .config import ConcurrentIngestionConfig
from .graph_reader import GraphReader
from .phases import (
    AdmittedUnit,
    ApplyResult,
    CommitResult,
    UnitLease,
    admit_bulk,
    admit_episode,
    analyze_bulk,
    analyze_episode,
    apply_change_set,
    commit_unit,
    enrich_unit,
    rebase_change_set,
)
from .scheduler import ConcurrentIngestor
from .sqlite_state_store import SQLiteIngestionStateStore
from .state_store import (
    Admission,
    AnalysisStart,
    CommitRecover,
    CommitTurn,
    CommitWait,
    DeltaUnit,
    IngestionStateStore,
    InMemoryIngestionStateStore,
)

__all__ = [
    'AddEpisodesToEdge',
    'AdmittedUnit',
    'Admission',
    'AnalysisStart',
    'ApplyResult',
    'ChangeSet',
    'ChangeSetRejected',
    'CommitRecover',
    'CommitResult',
    'CommitTurn',
    'CommitWait',
    'ConcurrentIngestionConfig',
    'ConcurrentIngestor',
    'DeltaUnit',
    'EndEdge',
    'GraphReader',
    'InMemoryIngestionStateStore',
    'IngestionStateStore',
    'PatchNode',
    'ResolutionTrace',
    'SQLiteIngestionStateStore',
    'SagaLink',
    'SetEdgeAttributes',
    'StaleCommitError',
    'UnitExpiredError',
    'UnitInProgressError',
    'UnitLease',
    'admit_bulk',
    'admit_episode',
    'analyze_bulk',
    'analyze_episode',
    'apply_change_set',
    'commit_unit',
    'enrich_unit',
    'rebase_change_set',
]
