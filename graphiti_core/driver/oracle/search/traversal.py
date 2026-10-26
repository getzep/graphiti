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

from __future__ import annotations

from typing import Literal

from ..config import GRAPH_IDENTIFIER
from ..errors import InvalidTraversalDepth


def validate_traversal_depth(max_depth: int) -> int:
    if isinstance(max_depth, bool) or not isinstance(max_depth, int) or not 1 <= max_depth <= 8:
        raise InvalidTraversalDepth('max_depth must be an integer between 1 and 8')
    return max_depth


def _validate_graph_name(graph_name: str) -> str:
    if not GRAPH_IDENTIFIER.fullmatch(graph_name):
        raise ValueError('graph_name must be a simple Oracle identifier')
    return graph_name.upper()


def build_bfs_sql(
    *,
    max_depth: int,
    graph_name: str = 'GRAPHITI_GRAPH',
    result: Literal['node', 'edge'] = 'node',
    where_sql: str = '1 = 1',
) -> tuple[str, dict[str, object]]:
    """Build a traversal with the only rendered value being a validated depth."""

    depth = validate_traversal_depth(max_depth)
    graph = _validate_graph_name(graph_name)
    pattern = f"""MATCH (origin IS graphiti_node)
        -[path_edge IS graphiti_edge
          WHERE path_edge.group_id = origin.group_id
            AND path_edge.kind IN ('RELATES_TO', 'MENTIONS')]->{{1,{depth}}}
        (reached IS graphiti_node)
      WHERE origin.group_id = reached.group_id"""

    if result == 'node':
        sql = f"""SELECT *
FROM (
  SELECT n.*
  FROM graphiti_nodes n
  JOIN (
    SELECT DISTINCT reachable.reached_uuid
    FROM GRAPH_TABLE (
      {graph}
      {pattern}
      COLUMNS (
        origin.uuid AS origin_uuid,
        reached.uuid AS reached_uuid,
        reached.kind AS reached_kind
      )
    ) reachable
    JOIN JSON_TABLE(
      :origin_uuids_json,
      '$[*]' COLUMNS (uuid VARCHAR2(36) PATH '$')
    ) origins ON origins.uuid = reachable.origin_uuid
    WHERE reachable.reached_kind = 'ENTITY'
  ) traversed ON traversed.reached_uuid = n.uuid
  WHERE n.kind = 'ENTITY'
    AND ({where_sql})
  ORDER BY n.uuid
)
WHERE ROWNUM <= :result_limit"""
        return sql, {}

    if result != 'edge':
        raise ValueError("result must be 'node' or 'edge'")
    sql = f"""SELECT *
FROM (
  SELECT e.*
  FROM graphiti_edges e
  JOIN (
    SELECT DISTINCT reachable.edge_uuid
    FROM GRAPH_TABLE (
      {graph}
      {pattern}
      ONE ROW PER STEP (step_source, step_edge, step_target)
      COLUMNS (
        origin.uuid AS origin_uuid,
        origin.group_id AS origin_group_id,
        step_edge.uuid AS edge_uuid,
        step_edge.kind AS edge_kind,
        step_edge.group_id AS edge_group_id
      )
    ) reachable
    JOIN JSON_TABLE(
      :origin_uuids_json,
      '$[*]' COLUMNS (uuid VARCHAR2(36) PATH '$')
    ) origins ON origins.uuid = reachable.origin_uuid
    WHERE reachable.edge_kind = 'RELATES_TO'
      AND reachable.edge_group_id = reachable.origin_group_id
  ) traversed ON traversed.edge_uuid = e.uuid
  JOIN graphiti_nodes src ON src.uuid = e.source_uuid
  JOIN graphiti_nodes dst ON dst.uuid = e.target_uuid
  WHERE e.kind = 'RELATES_TO'
    AND ({where_sql})
  ORDER BY e.uuid
)
WHERE ROWNUM <= :result_limit"""
    return sql, {}


def build_distance_neighbors_sql() -> str:
    """Fetch one hop per BFS round; never enumerate cyclic paths exponentially."""

    return """SELECT e.source_uuid AS from_uuid, e.target_uuid AS to_uuid
FROM JSON_TABLE(
  :frontier_uuids_json,
  '$[*]' COLUMNS (uuid VARCHAR2(36) PATH '$')
) f
JOIN graphiti_edges e ON e.source_uuid = f.uuid
JOIN graphiti_nodes center ON center.uuid = :center_node_uuid
JOIN graphiti_nodes reached ON reached.uuid = e.target_uuid
WHERE center.kind = 'ENTITY'
  AND (:scope_group_id IS NULL OR center.group_id = :scope_group_id)
  AND e.kind = 'RELATES_TO'
  AND e.group_id = center.group_id
  AND reached.kind = 'ENTITY'
  AND reached.group_id = center.group_id
UNION ALL
SELECT e.target_uuid AS from_uuid, e.source_uuid AS to_uuid
FROM JSON_TABLE(
  :frontier_uuids_json,
  '$[*]' COLUMNS (uuid VARCHAR2(36) PATH '$')
) f
JOIN graphiti_edges e ON e.target_uuid = f.uuid
JOIN graphiti_nodes center ON center.uuid = :center_node_uuid
JOIN graphiti_nodes reached ON reached.uuid = e.source_uuid
WHERE center.kind = 'ENTITY'
  AND (:scope_group_id IS NULL OR center.group_id = :scope_group_id)
  AND e.kind = 'RELATES_TO'
  AND e.group_id = center.group_id
  AND reached.kind = 'ENTITY'
  AND reached.group_id = center.group_id"""


def build_mentions_reranker_sql(graph_name: str = 'GRAPHITI_GRAPH') -> str:
    graph = _validate_graph_name(graph_name)
    return f"""SELECT mentions.candidate_uuid AS uuid,
       COUNT(DISTINCT episode_uuid) AS mention_count
FROM GRAPH_TABLE (
  {graph}
  MATCH (episode IS graphiti_node)
        -[mention_edge IS graphiti_edge]->
        (candidate IS graphiti_node)
  WHERE episode.kind = 'EPISODIC'
    AND mention_edge.kind = 'MENTIONS'
    AND candidate.kind = 'ENTITY'
    AND (:scope_group_id IS NULL OR episode.group_id = :scope_group_id)
    AND mention_edge.group_id = episode.group_id
    AND candidate.group_id = episode.group_id
  COLUMNS (
    episode.uuid AS episode_uuid,
    candidate.uuid AS candidate_uuid
  )
) mentions
JOIN JSON_TABLE(
  :candidate_uuids_json,
  '$[*]' COLUMNS (uuid VARCHAR2(36) PATH '$')
) candidates ON candidates.uuid = mentions.candidate_uuid
GROUP BY mentions.candidate_uuid
ORDER BY mention_count DESC, mentions.candidate_uuid"""
