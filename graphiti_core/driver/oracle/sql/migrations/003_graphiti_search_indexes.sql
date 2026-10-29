CREATE INDEX graphiti_nodes_group_kind_idx
ON graphiti_nodes(group_id, kind, uuid)
-- @statement
CREATE INDEX graphiti_edges_group_kind_idx
ON graphiti_edges(group_id, kind, uuid)
-- @statement
CREATE INDEX graphiti_edges_source_idx
ON graphiti_edges(source_uuid)
-- @statement
CREATE INDEX graphiti_edges_target_idx
ON graphiti_edges(target_uuid)
-- @statement
CREATE INDEX graphiti_nodes_text_idx
ON graphiti_nodes(search_text)
INDEXTYPE IS CTXSYS.CONTEXT
-- @statement
CREATE INDEX graphiti_edges_text_idx
ON graphiti_edges(search_text)
INDEXTYPE IS CTXSYS.CONTEXT
-- @statement
CREATE VECTOR INDEX graphiti_nodes_vector_idx
ON graphiti_nodes(name_embedding)
ORGANIZATION INMEMORY NEIGHBOR GRAPH
DISTANCE COSINE
WITH TARGET ACCURACY 95
-- @statement
CREATE VECTOR INDEX graphiti_edges_vector_idx
ON graphiti_edges(fact_embedding)
ORGANIZATION INMEMORY NEIGHBOR GRAPH
DISTANCE COSINE
WITH TARGET ACCURACY 95
