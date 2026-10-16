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


def label_propagation(projection: dict[str, list[Neighbor]]) -> list[list[str]]:
    # Implement the label propagation community detection algorithm.
    # 1. Start with each node being assigned its own community
    # 2. Each node will take on the community of the plurality of its neighbors
    # 3. Ties are broken by going to the largest community
    # 4. Continue until no communities change during propagation
    #
    # Termination is guaranteed by two measures (#402, #1355):
    # - Updates are applied in place while visiting nodes in a stable, sorted
    #   order. Synchronous snapshot updates let weight-symmetric structures
    #   (e.g. two nodes joined by parallel edges, edge_count >= 2) swap labels
    #   forever; with in-place updates the second node of such a pair already
    #   observes the first node's new label within the same pass and the
    #   oscillation stops. The sorted order also makes the result independent
    #   of the projection's insertion order.
    # - The loop is bounded by max(100, len(projection)) iterations. Chains
    #   need O(|V|) rounds to converge, so a fixed cap would truncate normally
    #   converging graphs; the cap therefore grows with the projection size.
    max_iterations = max(100, len(projection))
    node_order = sorted(projection.keys())
    community_map = {uuid: i for i, uuid in enumerate(node_order)}

    for _ in range(max_iterations):
        no_change = True

        for uuid in node_order:
            curr_community = community_map[uuid]

            community_candidates: dict[int, int] = defaultdict(int)
            for neighbor in projection[uuid]:
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
            break
    else:
        logger.warning(
            'label_propagation did not converge within %d iterations; returning best-effort '
            'communities',
            max_iterations,
        )

    community_cluster_map: dict[int, list[str]] = defaultdict(list)
    for uuid, community in community_map.items():
        community_cluster_map[community].append(uuid)

    return list(community_cluster_map.values())
