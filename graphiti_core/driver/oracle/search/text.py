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

import re
import unicodedata

from ..errors import InvalidSearchQuery
from .filters import OracleWhere

_SEARCH_TOKEN = re.compile(r'[^\W_]+', re.UNICODE)


def build_oracle_text_query(query: str, max_query_length: int = 8000) -> str:
    """Return a bounded any-term Oracle Text expression with cumulative ranking."""

    if max_query_length < 3:
        raise InvalidSearchQuery('max_query_length must be at least 3')
    normalized = unicodedata.normalize('NFKC', query)
    tokens = _SEARCH_TOKEN.findall(normalized)
    if not tokens:
        raise InvalidSearchQuery('search query has no searchable terms')

    parts: list[str] = []
    for token in tokens:
        part = '{' + token[: max_query_length - 2] + '}'
        candidate = ' ACCUM '.join([*parts, part])
        if len(candidate) > max_query_length:
            break
        parts.append(part)
    if not parts:
        raise InvalidSearchQuery('search query exceeds the configured limit')
    return ' ACCUM '.join(parts)


def build_node_fulltext_sql(*, kind: str, where: OracleWhere) -> str:
    if kind not in {'ENTITY', 'EPISODIC', 'COMMUNITY'}:
        raise ValueError(f'unsupported full-text node kind {kind}')
    return f"""SELECT ranked.*
FROM (
  SELECT n.*, SCORE(1) AS search_score
  FROM graphiti_nodes n
  WHERE n.kind = '{kind}'
    AND CONTAINS(n.search_text, :text_query, 1) > 0
    AND ({where.sql})
  ORDER BY SCORE(1) DESC, n.uuid
) ranked
WHERE ROWNUM <= :result_limit"""


def build_edge_fulltext_sql(*, where: OracleWhere) -> str:
    return f"""SELECT ranked.*
FROM (
  SELECT e.*, SCORE(1) AS search_score
  FROM graphiti_edges e
  JOIN graphiti_nodes src ON src.uuid = e.source_uuid
  JOIN graphiti_nodes dst ON dst.uuid = e.target_uuid
  WHERE e.kind = 'RELATES_TO'
    AND CONTAINS(e.search_text, :text_query, 1) > 0
    AND ({where.sql})
  ORDER BY SCORE(1) DESC, e.uuid
) ranked
WHERE ROWNUM <= :result_limit"""
