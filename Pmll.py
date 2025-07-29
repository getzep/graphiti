"""
PMLL memory wrapper around Graphiti – version 3.0
-------------------------------------------------
* Registers a minimalist Spatial ontology (SpatialNode + IsNear)
* Adds rich episodes; if `spatial_origin` is supplied it will
  • create / dedupe a SpatialNode
  • attach an IsNear edge to the previous waypoint (with distance_m)
* Simple hybrid RRF search helper (`query`)
"""

from __future__ import annotations

import asyncio
import json
import logging
import math
import uuid
from datetime import datetime, timezone
from typing import Dict, Optional, Tuple

from pydantic import Field, BaseModel

from graphiti_core import Graphiti
from graphiti_core.driver.neo4j_driver import Neo4jDriver
from graphiti_core.nodes import EpisodeType

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# --------------------------------------------------------------------------- #
# 1. Custom domain ontology                                                   #
# --------------------------------------------------------------------------- #
class SpatialNode(BaseModel):
    """A concrete point in 3-D Euclidean or geo space."""
    x: float = Field(..., description="X / longitude")
    y: float = Field(..., description="Y / latitude")
    z: float = Field(..., description="Z / altitude (m)")


class IsNear(BaseModel):
    """Spatial proximity relationship between two SpatialNodes."""
    distance_m: float = Field(..., description="Euclidean distance in metres")


ENTITY_TYPES: Dict[str, type[BaseModel]] = {"SpatialNode": SpatialNode}
EDGE_TYPES: Dict[str, type[BaseModel]] = {"IsNear": IsNear}
EDGE_TYPE_MAP = {("SpatialNode", "SpatialNode"): ["IsNear"]}

# --------------------------------------------------------------------------- #
# 2. PMLL wrapper                                                             #
# --------------------------------------------------------------------------- #
class PMLL:
    """Thin convenience layer that marries PMLL ideas to Graphiti."""

    _last_spatial: Optional[Tuple[str, float, float, float]] = None  # (uuid,x,y,z)

    def __init__(self, *, neo4j_uri: str, user: str, pwd: str):
        driver = Neo4jDriver(uri=neo4j_uri, user=user, password=pwd)
        self.graph = Graphiti(graph_driver=driver)

    # --------------------------- initialisation --------------------------- #
    async def init(self) -> None:
        """Create indices/constraints once per DB."""
        await self.graph.build_indices_and_constraints()  # Graphiti quick-start ✔

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

        # 1️⃣  Ingest the episode (Graphiti will extract regular entities)
        await self.graph.add_episode(
            name=f"ep@{datetime.now(timezone.utc).isoformat()}",
            episode_body=body,
            source=ep_type,
            source_description=description,
            reference_time=datetime.now(timezone.utc),
            group_id=group_id or "",
            entity_types=ENTITY_TYPES,      # ← custom ontology  [oai_citation:2‡Zep Documentation](https://help.getzep.com/graphiti/core-concepts/custom-entity-and-edge-types)
            edge_types=EDGE_TYPES,
            edge_type_map=EDGE_TYPE_MAP,
        )

        # 2️⃣  Spatial hook
        if spatial_origin is None:
            return

        x, y, z = spatial_origin

        # Simple in-memory dedupe of *consecutive* identical coords
        if self._last_spatial and self._last_spatial[1:] == spatial_origin:
            spatial_uuid = self._last_spatial[0]
        else:
            # Create a new SpatialNode
            spatial_node = SpatialNode(x=x, y=y, z=z)
            spatial_uuid = await self.graph.add_node(spatial_node)

            # Connect to previous waypoint if it exists
            if self._last_spatial:
                _, px, py, pz = self._last_spatial
                dist = math.dist((x, y, z), (px, py, pz))

                await self.graph.add_edge(
                    IsNear(distance_m=dist),
                    source_uuid=self._last_spatial[0],
                    target_uuid=spatial_uuid,
                )

        # Update cache
        self._last_spatial = (spatial_uuid, x, y, z)

    # ------------------------------- query -------------------------------- #
    async def query(
        self, question: str, centre_uuid: str | None = None, k: int = 5
    ):
        """Hybrid RRF search with optional centre-node re-ranking."""
        return await self.graph.search(question, center_node_uuid=centre_uuid, limit=k)

    # ------------------------------ cleanup ------------------------------- #
    async def close(self) -> None:
        await self.graph.close()


# --------------------------------------------------------------------------- #
# 3. Demo (run:  python -m pmll)                                              #
# --------------------------------------------------------------------------- #
async def _demo() -> None:
    pmll = PMLL(neo4j_uri="bolt://localhost:7687", user="neo4j", pwd="password")
    await pmll.init()

    await pmll.add_episode(
        "The robot entered Room A.",
        spatial_origin=(0, 0, 0),
        description="telemetry",
    )
    await pmll.add_episode(
        {"cmd": "move", "to": "Room B"},
        spatial_origin=(3, 4, 0),
        description="control instruction",
    )

    results = await pmll.query("Where is the robot now?")
    print("\nTop answers:")
    for r in results:
        print("•", r.fact)

    await pmll.close()


if __name__ == "__main__":
    asyncio.run(_demo())
