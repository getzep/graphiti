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
import random
from collections import defaultdict

from pydantic import BaseModel

logger = logging.getLogger(__name__)

MAX_ITERATIONS = 500


class Neighbor(BaseModel):
    node_uuid: str
    edge_count: int


def label_propagation(projection: dict[str, list[Neighbor]], seed: int = 0) -> list[list[str]]:
    community_map = {uuid: i for i, uuid in enumerate(projection.keys())}

    rng = random.Random(seed)

    for _ in range(MAX_ITERATIONS):
        no_change = True
        node_order = list(projection.keys())
        rng.shuffle(node_order)

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
            'Label propagation did not converge within %d iterations; '
            'using best-effort communities',
            MAX_ITERATIONS,
        )

    community_cluster_map: dict[int, list[str]] = defaultdict(list)
    for uuid, community in community_map.items():
        community_cluster_map[community].append(uuid)

    return list(community_cluster_map.values())
