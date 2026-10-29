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

from graphiti_core.driver.operations.episodic_edge_ops import EpisodicEdgeOperations
from graphiti_core.edges import EpisodicEdge

from .edge_base import StructuralEdgeOperationsMixin


class OracleEpisodicEdgeOperations(
    StructuralEdgeOperationsMixin[EpisodicEdge], EpisodicEdgeOperations
):
    kind = 'MENTIONS'
    source_kind = 'EPISODIC'
    target_kind = 'ENTITY'
