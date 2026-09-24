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

# Floor for the iteration budget, so tiny graphs still get room to converge.
MIN_LABEL_PROPAGATION_ITERATIONS = 10


class Neighbor(BaseModel):
    node_uuid: str
    edge_count: int


def label_propagation(projection: dict[str, list[Neighbor]]) -> list[list[str]]:
    """Label-propagation community detection.

    The loop is bounded: label propagation is not guaranteed to converge, and on
    balanced graph shapes the assignments can oscillate indefinitely. One
    relabelling pass per node is a generous budget — a run that has not settled
    by then is oscillating, not converging slowly — so on reaching the cap we
    log a warning and return the partition reached so far. That is a legitimate
    result for a heuristic algorithm; spinning forever is not.
    """
    community_map = {uuid: i for i, uuid in enumerate(projection.keys())}

    max_iterations = max(len(projection), MIN_LABEL_PROPAGATION_ITERATIONS)
    iteration = 0

    while True:
        iteration += 1
        if iteration > max_iterations:
            logger.warning(
                'label_propagation did not converge within %d iterations over %d nodes; '
                'returning the current partition',
                max_iterations,
                len(projection),
            )
            break

        no_change = True
        new_community_map: dict[str, int] = {}

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

            new_community_map[uuid] = new_community

            if new_community != curr_community:
                no_change = False

        if no_change:
            break

        community_map = new_community_map

    community_cluster_map: dict[int, list[str]] = defaultdict(list)
    for uuid, community in community_map.items():
        community_cluster_map[community].append(uuid)

    return list(community_cluster_map.values())
