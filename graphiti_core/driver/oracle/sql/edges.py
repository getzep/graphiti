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

VALIDATE_ENDPOINTS = """
SELECT uuid, kind, group_id
FROM graphiti_nodes
WHERE uuid IN (:source_uuid, :target_uuid)
ORDER BY uuid
FOR UPDATE
""".strip()

MERGE_EDGE = """
MERGE INTO graphiti_edges e
USING (SELECT :uuid AS uuid FROM dual) incoming
ON (e.uuid = incoming.uuid AND e.kind = :kind AND e.group_id = :group_id)
WHEN MATCHED THEN UPDATE SET
  e.source_uuid = :source_uuid,
  e.target_uuid = :target_uuid,
  e.name = :name,
  e.fact = :fact,
  e.episode_uuids = JSON(:episode_uuids),
  e.properties = JSON(:properties),
  e.search_text = :search_text,
  e.fact_embedding = :fact_embedding,
  e.created_at = :created_at,
  e.expired_at = :expired_at,
  e.valid_at = :valid_at,
  e.invalid_at = :invalid_at,
  e.reference_time = :reference_time
WHEN NOT MATCHED THEN INSERT (
  uuid, kind, source_uuid, target_uuid, group_id, name, fact, episode_uuids,
  properties, search_text, fact_embedding, created_at, expired_at, valid_at,
  invalid_at, reference_time
) VALUES (
  :uuid, :kind, :source_uuid, :target_uuid, :group_id, :name, :fact,
  JSON(:episode_uuids), JSON(:properties), :search_text, :fact_embedding,
  :created_at, :expired_at, :valid_at, :invalid_at, :reference_time
)
""".strip()

GET_EDGE_BY_UUID = """
SELECT * FROM graphiti_edges
WHERE uuid = :uuid AND kind = :kind
  AND (:scope_group_id IS NULL OR group_id = :scope_group_id)
""".strip()

GET_EDGES_BY_UUIDS = """
SELECT e.*
FROM graphiti_edges e
JOIN JSON_TABLE(:uuids_json, '$[*]' COLUMNS (uuid VARCHAR2(36) PATH '$')) wanted
  ON wanted.uuid = e.uuid
WHERE e.kind = :kind
  AND (:scope_group_id IS NULL OR e.group_id = :scope_group_id)
ORDER BY e.uuid
""".strip()

GET_EDGES_BY_GROUPS = """
SELECT * FROM (
  SELECT e.*
  FROM graphiti_edges e
  JOIN JSON_TABLE(:group_ids_json, '$[*]' COLUMNS (group_id VARCHAR2(255) PATH '$')) wanted
    ON wanted.group_id = e.group_id
  WHERE e.kind = :kind
    AND (:uuid_cursor IS NULL OR e.uuid < :uuid_cursor)
  ORDER BY e.uuid DESC
)
WHERE ROWNUM <= :result_limit
""".strip()

DELETE_EDGE = """
DELETE FROM graphiti_edges
WHERE uuid = :uuid AND kind = :kind
  AND (:scope_group_id IS NULL OR group_id = :scope_group_id)
""".strip()

DELETE_EDGES_BY_UUIDS = """
DELETE FROM graphiti_edges
WHERE kind = :kind
  AND (:scope_group_id IS NULL OR group_id = :scope_group_id)
  AND uuid IN (
    SELECT uuid FROM JSON_TABLE(
      :uuids_json,
      '$[*]' COLUMNS (uuid VARCHAR2(36) PATH '$')
    )
  )
""".strip()

GET_ENTITY_EDGES_BETWEEN = """
SELECT * FROM graphiti_edges
WHERE kind = 'RELATES_TO'
  AND (:scope_group_id IS NULL OR group_id = :scope_group_id)
  AND source_uuid = :source_uuid
  AND target_uuid = :target_uuid
ORDER BY uuid
""".strip()

GET_ENTITY_EDGES_BY_NODE = """
SELECT * FROM graphiti_edges
WHERE kind = 'RELATES_TO'
  AND (:scope_group_id IS NULL OR group_id = :scope_group_id)
  AND (source_uuid = :node_uuid OR target_uuid = :node_uuid)
ORDER BY uuid
""".strip()

GET_EDGE_EMBEDDING = """
SELECT fact_embedding FROM graphiti_edges
WHERE uuid = :uuid AND kind = 'RELATES_TO'
  AND (:scope_group_id IS NULL OR group_id = :scope_group_id)
""".strip()

GET_EDGE_SCOPE = """
SELECT kind, group_id
FROM graphiti_edges
WHERE uuid = :uuid
""".strip()
