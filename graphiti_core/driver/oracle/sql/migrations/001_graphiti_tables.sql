CREATE TABLE graphiti_nodes (
  uuid VARCHAR2(36 CHAR) PRIMARY KEY,
  kind VARCHAR2(32 CHAR) NOT NULL,
  group_id VARCHAR2(255 CHAR) NOT NULL,
  name VARCHAR2(1000 CHAR) NOT NULL,
  labels JSON,
  summary CLOB,
  source VARCHAR2(32 CHAR),
  source_description VARCHAR2(1000 CHAR),
  content CLOB,
  entity_edge_uuids JSON,
  properties JSON,
  search_text CLOB,
  name_embedding VECTOR(1024, FLOAT32),
  created_at TIMESTAMP WITH TIME ZONE NOT NULL,
  valid_at TIMESTAMP WITH TIME ZONE,
  first_episode_uuid VARCHAR2(36 CHAR),
  last_episode_uuid VARCHAR2(36 CHAR),
  last_summarized_at TIMESTAMP WITH TIME ZONE,
  last_summarized_episode_valid_at TIMESTAMP WITH TIME ZONE,
  CONSTRAINT graphiti_nodes_kind_ck
    CHECK (kind IN ('ENTITY', 'EPISODIC', 'COMMUNITY', 'SAGA'))
)
-- @statement
CREATE TABLE graphiti_edges (
  uuid VARCHAR2(36 CHAR) PRIMARY KEY,
  kind VARCHAR2(32 CHAR) NOT NULL,
  source_uuid VARCHAR2(36 CHAR) NOT NULL,
  target_uuid VARCHAR2(36 CHAR) NOT NULL,
  group_id VARCHAR2(255 CHAR) NOT NULL,
  name VARCHAR2(1000 CHAR),
  fact CLOB,
  episode_uuids JSON,
  properties JSON,
  search_text CLOB,
  fact_embedding VECTOR(1024, FLOAT32),
  created_at TIMESTAMP WITH TIME ZONE NOT NULL,
  expired_at TIMESTAMP WITH TIME ZONE,
  valid_at TIMESTAMP WITH TIME ZONE,
  invalid_at TIMESTAMP WITH TIME ZONE,
  reference_time TIMESTAMP WITH TIME ZONE,
  CONSTRAINT graphiti_edges_kind_ck
    CHECK (kind IN ('RELATES_TO', 'MENTIONS', 'HAS_MEMBER', 'HAS_EPISODE', 'NEXT_EPISODE')),
  CONSTRAINT graphiti_edges_source_fk
    FOREIGN KEY (source_uuid) REFERENCES graphiti_nodes(uuid),
  CONSTRAINT graphiti_edges_target_fk
    FOREIGN KEY (target_uuid) REFERENCES graphiti_nodes(uuid)
)
