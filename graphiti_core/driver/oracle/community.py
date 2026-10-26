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


def label_propagation(projection: dict[str, list[Neighbor]]) -> list[list[str]]:
    community_map = {uuid: i for i, uuid in enumerate(sorted(projection))}

    for _ in range(max(1, len(projection) * 4)):
        no_change = True

        for uuid in sorted(projection):
            neighbors = projection[uuid]
            curr_community = community_map[uuid]

            community_candidates: dict[int, int] = defaultdict(int)
            for neighbor in neighbors:
                if neighbor.node_uuid not in community_map:
                    continue
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
        # A weighted graph can still oscillate; a bounded connected-component
        # fallback keeps community construction finite and deterministic.
        seen: set[str] = set()
        components: list[list[str]] = []
        for start in sorted(projection):
            if start in seen:
                continue
            stack = [start]
            component: list[str] = []
            while stack:
                uuid = stack.pop()
                if uuid in seen:
                    continue
                seen.add(uuid)
                component.append(uuid)
                stack.extend(
                    neighbor.node_uuid
                    for neighbor in projection[uuid]
                    if neighbor.node_uuid in projection and neighbor.node_uuid not in seen
                )
            components.append(sorted(component))
        return components

    community_cluster_map: dict[int, list[str]] = defaultdict(list)
    for uuid, community in community_map.items():
        community_cluster_map[community].append(uuid)

    return list(community_cluster_map.values())
