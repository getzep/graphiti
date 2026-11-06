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

MIGRATION_READINESS = """
SELECT
  SUM(CASE WHEN state = 'APPLIED' THEN 1 ELSE 0 END) AS applied_count,
  SUM(CASE WHEN state = 'APPLYING' THEN 1 ELSE 0 END) AS applying_count
FROM graphiti_schema_migrations
""".strip()

TABLE_READINESS = """
SELECT COUNT(*) AS object_count
FROM user_tables
WHERE table_name IN ('GRAPHITI_NODES', 'GRAPHITI_EDGES')
""".strip()

PROPERTY_GRAPH_READINESS = """
SELECT COUNT(*) AS object_count
FROM user_property_graphs
WHERE graph_name = 'GRAPHITI_GRAPH'
""".strip()

TEXT_INDEX_READINESS = """
SELECT COUNT(*) AS object_count
FROM user_indexes
WHERE index_name IN ('GRAPHITI_NODES_TEXT_IDX', 'GRAPHITI_EDGES_TEXT_IDX')
""".strip()

VECTOR_INDEX_READINESS = """
SELECT COUNT(*) AS object_count
FROM user_indexes
WHERE index_name IN ('GRAPHITI_NODES_VECTOR_IDX', 'GRAPHITI_EDGES_VECTOR_IDX')
""".strip()

DROP_SEARCH_INDEXES = (
    'DROP INDEX graphiti_nodes_text_idx',
    'DROP INDEX graphiti_edges_text_idx',
    'DROP INDEX graphiti_nodes_vector_idx',
    'DROP INDEX graphiti_edges_vector_idx',
)
