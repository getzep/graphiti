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

from collections import defaultdict

from pydantic import BaseModel


class Neighbor(BaseModel):
    node_uuid: str
    edge_count: int


def label_propagation(
    projection: dict[str, list[Neighbor]], max_iterations: int = 100
) -> list[list[str]]:
    """Cluster nodes via the asynchronous label propagation algorithm.

    Each node iteratively adopts the most-weighted community label among
    its neighbours. Updates are applied in place (asynchronous LPA), so
    later nodes within the same iteration see earlier nodes' fresh
    labels. This avoids the oscillation that synchronous LPA exhibits on
    bipartite or weight-symmetric graphs, where reading from a snapshot
    and writing to a parallel map can flip-flop between two states
    indefinitely. ``max_iterations`` is a defensive upper bound that
    guarantees termination on any input.

    Ties between candidate communities of equal total weight are broken
    deterministically by the smallest community ID, so two runs over the
    same input produce the same clustering.
    """
    community_map: dict[str, int] = {uuid: i for i, uuid in enumerate(projection.keys())}

    for _ in range(max_iterations):
        no_change = True

        for uuid, neighbors in projection.items():
            community_candidates: dict[int, int] = defaultdict(int)
            for neighbor in neighbors:
                community_candidates[community_map[neighbor.node_uuid]] += neighbor.edge_count

            if not community_candidates:
                # A node with no neighbours has no candidate community to
                # adopt; leave it in its initial singleton community.
                continue

            new_community = min(
                community_candidates.items(),
                key=lambda item: (-item[1], item[0]),
            )[0]

            if new_community != community_map[uuid]:
                community_map[uuid] = new_community
                no_change = False

        if no_change:
            break

    community_cluster_map: dict[int, list[str]] = defaultdict(list)
    for uuid, community in community_map.items():
        community_cluster_map[community].append(uuid)

    return list(community_cluster_map.values())
