CREATE PROPERTY GRAPH graphiti_graph
  VERTEX TABLES (
    graphiti_nodes KEY (uuid)
      LABEL graphiti_node
      PROPERTIES (uuid, kind, group_id, name, created_at, valid_at)
  )
  EDGE TABLES (
    graphiti_edges KEY (uuid)
      SOURCE KEY (source_uuid) REFERENCES graphiti_nodes(uuid)
      DESTINATION KEY (target_uuid) REFERENCES graphiti_nodes(uuid)
      LABEL graphiti_edge
      PROPERTIES (uuid, kind, group_id, name, created_at, valid_at, invalid_at)
  )
