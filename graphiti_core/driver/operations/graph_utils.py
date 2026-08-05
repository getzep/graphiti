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

MAX_LABEL_PROPAGATION_ITERATIONS = 100

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
    # Nodes are processed in a fixed, deterministic order (sorted by uuid)
    # rather than projection's insertion order, which may vary run-to-run
    # since it is typically built from unordered DB query results. Because
    # updates are applied asynchronously (see below), the result is sensitive
    # to processing order, so a stable order is required for reproducible
    # community assignments across runs on the same underlying graph (#1355).
    node_order = sorted(projection.keys())
    community_map = {uuid: i for i, uuid in enumerate(node_order)}

    # Updates are applied asynchronously (in place), not from a single prior
    # snapshot. A synchronous update lets two nodes joined by edge_count >= 2
    # swap communities with each other every round forever, since both compute
    # their new label from the other's *old* label at the same time (#1355).
    # Updating in place means the second node in a pair already observes the
    # first node's new label within the same pass, which converges instead of
    # oscillating.
    for _ in range(MAX_LABEL_PROPAGATION_ITERATIONS):
        no_change = True

        for uuid in node_order:
            neighbors = projection[uuid]
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
            break
    else:
        logger.warning(
            'label_propagation did not converge after %d iterations; '
            'returning best-effort communities',
            MAX_LABEL_PROPAGATION_ITERATIONS,
        )

    community_cluster_map: dict[int, list[str]] = defaultdict(list)
    for uuid, community in community_map.items():
        community_cluster_map[community].append(uuid)

    return list(community_cluster_map.values())
