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

CLEAR_ALL_EDGES = 'DELETE FROM graphiti_edges'
CLEAR_ALL_NODES = 'DELETE FROM graphiti_nodes'

CLEAR_GROUP_EDGES = """
DELETE FROM graphiti_edges
WHERE group_id IN (
  SELECT group_id FROM JSON_TABLE(
    :group_ids_json,
    '$[*]' COLUMNS (group_id VARCHAR2(255) PATH '$')
  )
)
""".strip()

CLEAR_GROUP_NODES = """
DELETE FROM graphiti_nodes
WHERE group_id IN (
  SELECT group_id FROM JSON_TABLE(
    :group_ids_json,
    '$[*]' COLUMNS (group_id VARCHAR2(255) PATH '$')
  )
)
""".strip()

GET_GROUP_IDS = """
SELECT DISTINCT group_id FROM graphiti_nodes
WHERE kind = 'ENTITY'
ORDER BY group_id
""".strip()

GET_GROUP_ENTITIES = """
SELECT * FROM graphiti_nodes
WHERE kind = 'ENTITY' AND group_id = :group_id
ORDER BY uuid
""".strip()

GET_GROUP_RELATIONS = """
SELECT source_uuid, target_uuid, COUNT(*) AS edge_count
FROM graphiti_edges
WHERE kind = 'RELATES_TO' AND group_id = :group_id
GROUP BY source_uuid, target_uuid
""".strip()

GET_MENTIONED_NODES = """
SELECT n.*
FROM graphiti_nodes n
WHERE n.kind = 'ENTITY'
  AND (:scope_group_id IS NULL OR n.group_id = :scope_group_id)
  AND EXISTS (
    SELECT 1
    FROM graphiti_edges e
    JOIN JSON_TABLE(:episode_uuids, '$[*]' COLUMNS (uuid VARCHAR2(36) PATH '$')) wanted
      ON wanted.uuid = e.source_uuid
    WHERE e.target_uuid = n.uuid AND e.kind = 'MENTIONS'
      AND (:scope_group_id IS NULL OR e.group_id = :scope_group_id)
  )
ORDER BY n.uuid
""".strip()

GET_COMMUNITIES_BY_NODES = """
SELECT c.*
FROM graphiti_nodes c
WHERE c.kind = 'COMMUNITY'
  AND (:scope_group_id IS NULL OR c.group_id = :scope_group_id)
  AND EXISTS (
    SELECT 1
    FROM graphiti_edges e
    JOIN JSON_TABLE(:node_uuids, '$[*]' COLUMNS (uuid VARCHAR2(36) PATH '$')) wanted
      ON wanted.uuid = e.target_uuid
    WHERE e.source_uuid = c.uuid AND e.kind = 'HAS_MEMBER'
      AND (:scope_group_id IS NULL OR e.group_id = :scope_group_id)
  )
ORDER BY c.uuid
""".strip()

GET_EXISTING_ENTITY_COMMUNITY = """
SELECT c.*
FROM graphiti_nodes c
JOIN graphiti_edges e ON e.source_uuid = c.uuid
WHERE c.kind = 'COMMUNITY'
  AND e.kind = 'HAS_MEMBER'
  AND e.target_uuid = :entity_uuid
  AND (:scope_group_id IS NULL OR c.group_id = :scope_group_id)
  AND (:scope_group_id IS NULL OR e.group_id = :scope_group_id)
FETCH FIRST 1 ROW ONLY
""".strip()

GET_NEIGHBOR_COMMUNITIES = """
SELECT c.*
FROM graphiti_nodes c
WHERE c.uuid = (
  SELECT member.source_uuid
  FROM graphiti_edges member
  JOIN graphiti_nodes community
    ON community.uuid = member.source_uuid AND community.kind = 'COMMUNITY'
  JOIN graphiti_edges relation
    ON relation.source_uuid = member.target_uuid OR relation.target_uuid = member.target_uuid
  WHERE member.kind = 'HAS_MEMBER'
    AND relation.kind = 'RELATES_TO'
    AND (:scope_group_id IS NULL OR member.group_id = :scope_group_id)
    AND (:scope_group_id IS NULL OR relation.group_id = :scope_group_id)
    AND (relation.source_uuid = :entity_uuid OR relation.target_uuid = :entity_uuid)
  GROUP BY member.source_uuid
  ORDER BY COUNT(*) DESC, member.source_uuid
  FETCH FIRST 1 ROW ONLY
)
  AND (:scope_group_id IS NULL OR c.group_id = :scope_group_id)
""".strip()

REMOVE_ALL_COMMUNITIES = """
DELETE FROM graphiti_nodes WHERE kind = 'COMMUNITY'
""".strip()

REMOVE_GROUP_COMMUNITIES = """
DELETE FROM graphiti_nodes
WHERE kind = 'COMMUNITY'
  AND group_id IN (
    SELECT group_id FROM JSON_TABLE(
      :group_ids_json,
      '$[*]' COLUMNS (group_id VARCHAR2(255) PATH '$')
    )
  )
""".strip()
