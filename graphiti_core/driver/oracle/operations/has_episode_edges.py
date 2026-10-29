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

from graphiti_core.driver.operations.has_episode_edge_ops import HasEpisodeEdgeOperations
from graphiti_core.edges import HasEpisodeEdge

from .edge_base import StructuralEdgeOperationsMixin


class OracleHasEpisodeEdgeOperations(
    StructuralEdgeOperationsMixin[HasEpisodeEdge], HasEpisodeEdgeOperations
):
    kind = 'HAS_EPISODE'
    source_kind = 'SAGA'
    target_kind = 'EPISODIC'
