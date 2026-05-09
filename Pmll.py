# Copyright 2025, Zep Software, Inc.
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

"""
PMLL memory wrapper around Graphiti – version 3.2
-------------------------------------------------
* Registers SpatialNode / IsNear ontology
* Adds episodes with optional spatial anchors & distance-chaining
* Hybrid RRF search helper
"""

from __future__ import annotations

import asyncio
import json
import logging
import math
from datetime import datetime, timezone
from typing import Dict, Optional, Tuple

from pydantic import BaseModel, Field

from graphiti_core import Graphiti
from graphiti_core.driver.neo4j_driver import Neo4jDriver
from graphiti_core.nodes import EpisodeType

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# --------------------------------------------------------------------------- #
# 1. Custom ontology                                                          #
# --------------------------------------------------------------------------- #
class SpatialNode(BaseModel):
    """Cartesian/geo point in 3-D space."""
    x: float = Field(..., description="X / longitude")
    y: float = Field(..., description="Y / latitude")
    z: float = Field(..., description="Z / altitude (m)")


class IsNear(BaseModel):
    """Proximity relation between two SpatialNodes."""
    distance_m: float = Field(..., description="Euclidean distance in metres")


ENTITY_TYPES: Dict[str, type[BaseModel]] = {"SpatialNode": SpatialNode}
EDGE_TYPES: Dict[str, type[BaseModel]] = {"IsNear": IsNear}
EDGE_TYPE_MAP = {("SpatialNode", "SpatialNode"): ["IsNear"]}

# --------------------------------------------------------------------------- #
# 2. PMLL wrapper                                                             #
# --------------------------------------------------------------------------- #
class PMLL:
    """Thin convenience layer that marries PMLL ideas to Graphiti."""

    def __init__(self, *, neo4j_uri: str, user: str, pwd: str):
        driver = Neo4jDriver(uri=neo4j_uri, user=user, password=pwd)
        self.graph = Graphiti(graph_driver=driver)
        self._last_spatial: Optional[Tuple[str, float, float, float]] = None  # uuid,x,y,z

    # --------------------------- initialisation --------------------------- #
    async def init(self) -> None:
        """Create indices/constraints once per DB."""
        await self.graph.build_indices_and_constraints()

    # ------------------------------- ingest ------------------------------- #
    async def add_episode(
        self,
        content: str | dict,
        *,
        spatial_origin: Tuple[float, float, float] | None = None,
        description: str = "",
        group_id: str | None = None,
    ) -> None:
        """Persist raw experience (+ optional spatial anchor)."""
        ep_type = EpisodeType.text if isinstance(content, str) else EpisodeType.json
        body = content if isinstance(content, str) else json.dumps(content)

        await self.graph.add_episode(
            name=f"ep@{datetime.now(timezone.utc).isoformat()}",
            episode_body=body,
            source=ep_type,
            source_description=description,
            reference_time=datetime.now(timezone.utc),
            group_id=group_id or "",
            entity_types=ENTITY_TYPES,
            edge_types=EDGE_TYPES,
            edge_type_map=EDGE_TYPE_MAP,
        )

        if spatial_origin is None:
            return

        x, y, z = spatial_origin

        # Re-use node if identical to previous coords
        if self._last_spatial and self._last_spatial[1:] == spatial_origin:
            spatial_uuid = self._last_spatial[0]
        else:
            spatial_uuid = await self.graph.add_node(SpatialNode(x=x, y=y, z=z))

            # Connect to previous waypoint
            if self._last_spatial:
                _, px, py, pz = self._last_spatial
                await self.graph.add_edge(
                    IsNear(distance_m=math.dist((x, y, z), (px, py, pz))),
                    source_uuid=self._last_spatial[0],
                    target_uuid=spatial_uuid,
                )

        self._last_spatial = (spatial_uuid, x, y, z)

    # ------------------------------- query -------------------------------- #
    async def query(
        self, question: str, center_uuid: str | None = None, k: int = 5
    ):
        """Hybrid RRF search with optional centre-node re-ranking."""
        return await self.graph.search(question, center_node_uuid=center_uuid, limit=k)

    # ------------------------------ cleanup ------------------------------- #
    async def close(self) -> None:
        await self.graph.close()


# --------------------------------------------------------------------------- #
# 3. Demo                                                                     #
# --------------------------------------------------------------------------- #
async def _demo() -> None:
    pmll = PMLL(neo4j_uri="bolt://localhost:7687", user="neo4j", pwd="password")
    await pmll.init()

    await pmll.add_episode("Robot entered Room A.", spatial_origin=(0, 0, 0))
    await pmll.add_episode({"cmd": "move", "to": "Room B"}, spatial_origin=(3, 4, 0))

    for hit in await pmll.query("Where is the robot now?"):
        print("→", hit.fact)

    await pmll.close()


if __name__ == "__main__":
    asyncio.run(_demo())
