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

from .filters import OracleWhere


def build_node_vector_sql(*, kind: str, where: OracleWhere) -> str:
    if kind not in {'ENTITY', 'COMMUNITY'}:
        raise ValueError(f'unsupported vector node kind {kind}')
    return f"""SELECT ranked.*
FROM (
  SELECT n.*,
         1 - VECTOR_DISTANCE(n.name_embedding, :search_vector, COSINE) AS similarity
  FROM graphiti_nodes n
  WHERE n.kind = '{kind}'
    AND n.name_embedding IS NOT NULL
    AND ({where.sql})
  ORDER BY VECTOR_DISTANCE(n.name_embedding, :search_vector, COSINE), n.uuid
) ranked
WHERE ranked.similarity >= :min_score
  AND ROWNUM <= :result_limit"""


def build_edge_vector_sql(*, where: OracleWhere, endpoint_sql: str = '1 = 1') -> str:
    return f"""SELECT ranked.*
FROM (
  SELECT e.*,
         1 - VECTOR_DISTANCE(e.fact_embedding, :search_vector, COSINE) AS similarity
  FROM graphiti_edges e
  JOIN graphiti_nodes src ON src.uuid = e.source_uuid
  JOIN graphiti_nodes dst ON dst.uuid = e.target_uuid
  WHERE e.kind = 'RELATES_TO'
    AND e.fact_embedding IS NOT NULL
    AND ({where.sql})
    AND ({endpoint_sql})
  ORDER BY VECTOR_DISTANCE(e.fact_embedding, :search_vector, COSINE), e.uuid
) ranked
WHERE ranked.similarity >= :min_score
  AND ROWNUM <= :result_limit"""


COMMUNITY_EMBEDDINGS_SQL = """
SELECT n.uuid, n.name_embedding
FROM graphiti_nodes n
WHERE n.kind = 'COMMUNITY'
  AND (:scope_group_id IS NULL OR n.group_id = :scope_group_id)
  AND n.uuid IN (
    SELECT value
    FROM JSON_TABLE(
      :community_uuids_json,
      '$[*]' COLUMNS (value VARCHAR2(36) PATH '$')
    )
  )
  AND n.name_embedding IS NOT NULL
ORDER BY n.uuid
""".strip()
