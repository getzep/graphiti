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

MERGE_ENTITY_NODE = """
MERGE INTO graphiti_nodes n
USING (SELECT :uuid AS uuid FROM dual) incoming
ON (n.uuid = incoming.uuid AND n.kind = :kind AND n.group_id = :group_id)
WHEN MATCHED THEN UPDATE SET
  n.name = :name,
  n.labels = JSON(:labels),
  n.summary = :summary,
  n.properties = JSON(:properties),
  n.search_text = :search_text,
  n.name_embedding = :name_embedding,
  n.created_at = :created_at
WHEN NOT MATCHED THEN INSERT (
  uuid, kind, group_id, name, labels, summary, properties,
  search_text, name_embedding, created_at
) VALUES (
  :uuid, :kind, :group_id, :name, JSON(:labels), :summary, JSON(:properties),
  :search_text, :name_embedding, :created_at
)
""".strip()

MERGE_EPISODE_NODE = """
MERGE INTO graphiti_nodes n
USING (SELECT :uuid AS uuid FROM dual) incoming
ON (n.uuid = incoming.uuid AND n.kind = :kind AND n.group_id = :group_id)
WHEN MATCHED THEN UPDATE SET
  n.name = :name,
  n.labels = JSON(:labels),
  n.source = :source,
  n.source_description = :source_description,
  n.content = :content,
  n.entity_edge_uuids = JSON(:entity_edge_uuids),
  n.properties = JSON(:properties),
  n.search_text = :search_text,
  n.created_at = :created_at,
  n.valid_at = :valid_at
WHEN NOT MATCHED THEN INSERT (
  uuid, kind, group_id, name, labels, source, source_description, content,
  entity_edge_uuids, properties, search_text, created_at, valid_at
) VALUES (
  :uuid, :kind, :group_id, :name, JSON(:labels), :source,
  :source_description, :content, JSON(:entity_edge_uuids), JSON(:properties),
  :search_text, :created_at, :valid_at
)
""".strip()

GET_ENTITY_BY_UUID = """
SELECT * FROM graphiti_nodes
WHERE uuid = :uuid AND kind = 'ENTITY'
  AND (:scope_group_id IS NULL OR group_id = :scope_group_id)
""".strip()

GET_EPISODE_BY_UUID = """
SELECT * FROM graphiti_nodes
WHERE uuid = :uuid AND kind = 'EPISODIC'
  AND (:scope_group_id IS NULL OR group_id = :scope_group_id)
""".strip()

GET_ENTITIES_BY_UUIDS = """
SELECT n.*
FROM graphiti_nodes n
JOIN JSON_TABLE(:uuids_json, '$[*]' COLUMNS (uuid VARCHAR2(36) PATH '$')) wanted
  ON wanted.uuid = n.uuid
WHERE n.kind = 'ENTITY'
  AND (:scope_group_id IS NULL OR n.group_id = :scope_group_id)
ORDER BY n.uuid
""".strip()

GET_EPISODES_BY_UUIDS = GET_ENTITIES_BY_UUIDS.replace("'ENTITY'", "'EPISODIC'")

GET_ENTITIES_BY_NAMES = """
SELECT n.*
FROM graphiti_nodes n
JOIN JSON_TABLE(:names_json, '$[*]' COLUMNS (name VARCHAR2(4000) PATH '$')) wanted_name
  ON wanted_name.name = n.name
JOIN JSON_TABLE(
  :group_ids_json,
  '$[*]' COLUMNS (group_id VARCHAR2(255) PATH '$')
) wanted_group ON wanted_group.group_id = n.group_id
WHERE n.kind = 'ENTITY'
ORDER BY n.uuid
""".strip()

GET_ENTITIES_BY_ATTRIBUTES = """
SELECT * FROM (
  SELECT n.*
  FROM graphiti_nodes n
  WHERE n.kind = 'ENTITY'
    AND {where_sql}
  ORDER BY n.uuid
)
WHERE ROWNUM <= :result_limit
""".strip()

GET_ENTITIES_BY_GROUPS = """
SELECT * FROM (
  SELECT n.*
  FROM graphiti_nodes n
  JOIN JSON_TABLE(:group_ids_json, '$[*]' COLUMNS (group_id VARCHAR2(255) PATH '$')) wanted
    ON wanted.group_id = n.group_id
  WHERE n.kind = 'ENTITY'
    AND (:uuid_cursor IS NULL OR n.uuid < :uuid_cursor)
  ORDER BY n.uuid DESC
)
WHERE ROWNUM <= :result_limit
""".strip()

GET_EPISODES_BY_GROUPS = GET_ENTITIES_BY_GROUPS.replace("'ENTITY'", "'EPISODIC'")

DELETE_NODE_EDGES = """
DELETE FROM graphiti_edges
WHERE (source_uuid = :uuid OR target_uuid = :uuid)
  AND (:scope_group_id IS NULL OR group_id = :scope_group_id)
""".strip()

DELETE_NODE = """
DELETE FROM graphiti_nodes
WHERE uuid = :uuid AND kind = :kind
  AND (:scope_group_id IS NULL OR group_id = :scope_group_id)
""".strip()

DELETE_GROUP_EDGES = 'DELETE FROM graphiti_edges WHERE group_id = :group_id'

DELETE_GROUP_NODES = 'DELETE FROM graphiti_nodes WHERE group_id = :group_id AND kind = :kind'

DELETE_NODES_BY_UUIDS = """
DELETE FROM graphiti_nodes
WHERE kind = :kind
  AND (:scope_group_id IS NULL OR group_id = :scope_group_id)
  AND uuid IN (
    SELECT uuid FROM JSON_TABLE(
      :uuids_json,
      '$[*]' COLUMNS (uuid VARCHAR2(36) PATH '$')
    )
  )
""".strip()

GET_ENTITY_EMBEDDING = """
SELECT name_embedding FROM graphiti_nodes
WHERE uuid = :uuid AND kind = 'ENTITY'
  AND (:scope_group_id IS NULL OR group_id = :scope_group_id)
""".strip()

GET_EPISODES_BY_ENTITY = """
SELECT n.*
FROM graphiti_nodes n
WHERE n.kind = 'EPISODIC'
  AND (:scope_group_id IS NULL OR n.group_id = :scope_group_id)
  AND EXISTS (
    SELECT 1
    FROM graphiti_edges e
    WHERE e.source_uuid = n.uuid
      AND e.kind = 'MENTIONS'
      AND e.target_uuid = :entity_node_uuid
      AND (:scope_group_id IS NULL OR e.group_id = :scope_group_id)
  )
ORDER BY n.valid_at, n.created_at
""".strip()

RETRIEVE_EPISODES = """
SELECT * FROM (
  SELECT n.*
  FROM graphiti_nodes n
  WHERE n.kind = 'EPISODIC'
    AND n.valid_at <= :reference_time
    AND (:all_groups = 1 OR n.group_id IN (
      SELECT group_id FROM JSON_TABLE(
        :group_ids_json,
        '$[*]' COLUMNS (group_id VARCHAR2(255) PATH '$')
      )
    ))
    AND (:source IS NULL OR n.source = :source)
    AND (:saga IS NULL OR EXISTS (
      SELECT 1
      FROM graphiti_edges h
      JOIN graphiti_nodes s ON s.uuid = h.source_uuid
      WHERE h.kind = 'HAS_EPISODE'
        AND h.target_uuid = n.uuid
        AND s.kind = 'SAGA'
        AND s.name = :saga
    ))
  ORDER BY n.valid_at DESC, n.created_at DESC
)
WHERE ROWNUM <= :last_n
ORDER BY valid_at ASC, created_at ASC
""".strip()

MERGE_COMMUNITY_NODE = """
MERGE INTO graphiti_nodes n
USING (SELECT :uuid AS uuid FROM dual) incoming
ON (n.uuid = incoming.uuid AND n.kind = :kind AND n.group_id = :group_id)
WHEN MATCHED THEN UPDATE SET
  n.name = :name,
  n.labels = JSON(:labels),
  n.summary = :summary,
  n.search_text = :search_text,
  n.name_embedding = :name_embedding,
  n.created_at = :created_at
WHEN NOT MATCHED THEN INSERT (
  uuid, kind, group_id, name, labels, summary, search_text, name_embedding, created_at
) VALUES (
  :uuid, :kind, :group_id, :name, JSON(:labels), :summary,
  :search_text, :name_embedding, :created_at
)
""".strip()

MERGE_SAGA_NODE = """
MERGE INTO graphiti_nodes n
USING (SELECT :uuid AS uuid FROM dual) incoming
ON (n.uuid = incoming.uuid AND n.kind = :kind AND n.group_id = :group_id)
WHEN MATCHED THEN UPDATE SET
  n.name = :name,
  n.labels = JSON(:labels),
  n.summary = :summary,
  n.search_text = :search_text,
  n.created_at = :created_at,
  n.first_episode_uuid = :first_episode_uuid,
  n.last_episode_uuid = :last_episode_uuid,
  n.last_summarized_at = :last_summarized_at,
  n.last_summarized_episode_uuid = :last_summarized_episode_uuid,
  n.last_summarized_episode_valid_at = :last_summarized_episode_valid_at
WHEN NOT MATCHED THEN INSERT (
  uuid, kind, group_id, name, labels, summary, search_text, created_at,
  first_episode_uuid, last_episode_uuid, last_summarized_at,
  last_summarized_episode_uuid, last_summarized_episode_valid_at
) VALUES (
  :uuid, :kind, :group_id, :name, JSON(:labels), :summary, :search_text, :created_at,
  :first_episode_uuid, :last_episode_uuid, :last_summarized_at,
  :last_summarized_episode_uuid, :last_summarized_episode_valid_at
)
""".strip()

GET_COMMUNITY_BY_UUID = """
SELECT * FROM graphiti_nodes
WHERE uuid = :uuid AND kind = 'COMMUNITY'
  AND (:scope_group_id IS NULL OR group_id = :scope_group_id)
""".strip()

GET_SAGA_BY_UUID = """
SELECT * FROM graphiti_nodes
WHERE uuid = :uuid AND kind = 'SAGA'
  AND (:scope_group_id IS NULL OR group_id = :scope_group_id)
""".strip()

GET_COMMUNITIES_BY_UUIDS = GET_ENTITIES_BY_UUIDS.replace("'ENTITY'", "'COMMUNITY'")
GET_SAGAS_BY_UUIDS = GET_ENTITIES_BY_UUIDS.replace("'ENTITY'", "'SAGA'")
GET_COMMUNITIES_BY_GROUPS = GET_ENTITIES_BY_GROUPS.replace("'ENTITY'", "'COMMUNITY'")
GET_SAGAS_BY_GROUPS = GET_ENTITIES_BY_GROUPS.replace("'ENTITY'", "'SAGA'")

GET_COMMUNITY_EMBEDDING = """
SELECT name_embedding FROM graphiti_nodes
WHERE uuid = :uuid AND kind = 'COMMUNITY'
  AND (:scope_group_id IS NULL OR group_id = :scope_group_id)
""".strip()

GET_NODE_SCOPE = """
SELECT kind, group_id
FROM graphiti_nodes
WHERE uuid = :uuid
""".strip()

GET_PREVIOUS_SAGA_EPISODE = """
SELECT e.uuid
FROM graphiti_nodes e
JOIN graphiti_edges h ON h.target_uuid = e.uuid
WHERE h.kind = 'HAS_EPISODE'
  AND h.source_uuid = :saga_uuid
  AND (:scope_group_id IS NULL OR h.group_id = :scope_group_id)
  AND e.kind = 'EPISODIC'
  AND (:scope_group_id IS NULL OR e.group_id = :scope_group_id)
  AND e.uuid <> :current_episode_uuid
ORDER BY e.valid_at DESC, e.created_at DESC
FETCH FIRST 1 ROW ONLY
""".strip()

GET_SAGA_EPISODE_CONTENTS = """
SELECT * FROM (
  SELECT e.uuid, e.content, e.valid_at, e.created_at
  FROM graphiti_nodes e
  JOIN graphiti_edges h ON h.target_uuid = e.uuid
  WHERE h.kind = 'HAS_EPISODE'
    AND h.source_uuid = :saga_uuid
    AND (:scope_group_id IS NULL OR h.group_id = :scope_group_id)
    AND e.kind = 'EPISODIC'
    AND (:scope_group_id IS NULL OR e.group_id = :scope_group_id)
    AND e.created_at IS NOT NULL
    AND (
      :since IS NULL
      OR e.created_at > :since
      OR (e.created_at = :since AND (:after_uuid IS NULL OR e.uuid > :after_uuid))
    )
  ORDER BY e.created_at ASC, e.uuid ASC
)
WHERE ROWNUM <= :result_limit
""".strip()
