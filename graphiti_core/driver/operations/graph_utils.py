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

import logging
from collections import defaultdict

from pydantic import BaseModel

logger = logging.getLogger(__name__)


class Neighbor(BaseModel):
    node_uuid: str
    edge_count: int


# Lower bound on the number of sweeps. A label moves at most one hop per sweep against the
# iteration order, so a connected component can legitimately need about as many sweeps as it
# has nodes; the default cap therefore also scales with the projection size.
MAX_LABEL_PROPAGATION_SWEEPS = 100


def label_propagation(
    projection: dict[str, list[Neighbor]], max_sweeps: int | None = None
) -> list[list[str]]:
    """Cluster the projection with asynchronous label propagation.

    Every node starts in its own community and repeatedly adopts the community that carries
    the most edge weight among its neighbours, ties going to the larger community id, until
    a full sweep changes nothing. Labels are updated in place during the sweep so that later
    nodes see earlier moves; a synchronous update never terminates when two nodes are joined
    only by parallel edges, because each adopts the other's label on every sweep.

    ``max_sweeps`` is a hard termination guarantee for any remaining oscillation. It defaults
    to ``max(MAX_LABEL_PROPAGATION_SWEEPS, len(projection))`` so that chains, which converge
    one hop per sweep, are never cut short. The current partition is returned, and a warning
    logged, if the cap is reached before a fixed point.
    """
    if max_sweeps is None:
        max_sweeps = max(MAX_LABEL_PROPAGATION_SWEEPS, len(projection))

    community_map = {uuid: i for i, uuid in enumerate(projection.keys())}

    converged = False
    for _ in range(max_sweeps):
        no_change = True

        for uuid, neighbors in projection.items():
            curr_community = community_map[uuid]

            community_candidates: dict[int, int] = defaultdict(int)
            for neighbor in neighbors:
                community_candidates[community_map[neighbor.node_uuid]] += neighbor.edge_count
            community_lst = [
                (count, community) for community, count in community_candidates.items()
            ]

            community_lst.sort(reverse=True)
            candidate_rank, community_candidate = community_lst[0] if community_lst else (0, -1)
            if community_candidate != -1 and candidate_rank > 1:
                new_community = community_candidate
            else:
                new_community = max(community_candidate, curr_community)

            if new_community != curr_community:
                community_map[uuid] = new_community
                no_change = False

        if no_change:
            converged = True
            break

    if not converged:
        logger.warning(
            'label_propagation stopped after %d sweeps without converging on %d nodes; '
            'returning the current partition',
            max_sweeps,
            len(projection),
        )

    community_cluster_map: dict[int, list[str]] = defaultdict(list)
    for uuid, community in community_map.items():
        community_cluster_map[community].append(uuid)

    return list(community_cluster_map.values())
