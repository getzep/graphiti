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

"""Narrow Cypher query translations for embedded Relate compatibility.

Relate executes most Graphiti Cypher verbatim, with four targeted incompatibilities
identified in SPIKE-001 and transaction semantics:
1. Neo4j vector-property procedures (``db.create.setNodeVectorProperty`` and
   ``db.create.setRelationshipVectorProperty``) do not exist; direct property
   assignment preserves cosine vector similarity searchability.
2. Fulltext search procedures accept two arguments (index name and query) rather
   than three (``{limit: $limit}``), relying on trailing ``ORDER BY ... LIMIT``.
3. ``CALL (n) { DETACH DELETE n } IN TRANSACTIONS`` is rejected inside explicit
   transactions; ``RelateStore`` executes inside explicit transactions.
4. Variable-length traversals above Relate's default 500-hop safety limit fail with
   ``TraversalDepthExceeded``; hop bounds are capped at ``RELATE_MAX_BFS_DEPTH``.
"""

from __future__ import annotations

import re

RELATE_MAX_BFS_DEPTH = 500

_VECTOR_PROC_RE = re.compile(
    r'WITH\s+(?P<var>[A-Za-z_][A-Za-z0-9_]*)'
    r'(?:\s*,\s*[A-Za-z_][A-Za-z0-9_]*)?\s+'
    r'CALL\s+db\.create\.set(?:Node|Relationship)VectorProperty\(\s*'
    r'(?P=var)\s*,\s*["\'](?P<prop>[A-Za-z_][A-Za-z0-9_]*)["\']\s*,\s*'
    r'(?P<expr>[^)]+?)\s*\)',
    re.IGNORECASE,
)

_FT_ARG = r'(?:"[^"]*"|\'[^\']*\'|\$[A-Za-z_][A-Za-z0-9_]*)'
_FULLTEXT_PROC_RE = re.compile(
    rf'(CALL\s+db\.index\.fulltext\.query(?:Nodes|Relationships)\(\s*{_FT_ARG}\s*,\s*{_FT_ARG})'
    r'\s*,\s*\{[^}]*\}\s*\)',
    re.IGNORECASE,
)

_CALL_IN_TX_DELETE_RE = re.compile(
    r'CALL\s*\(\s*(?P<var>[A-Za-z_][A-Za-z0-9_]*)\s*\)\s*\{\s*'
    r'DETACH\s+DELETE\s+(?P=var)\s*\}\s*IN\s+TRANSACTIONS'
    r'(?:\s+OF\s+(?:\$[A-Za-z_][A-Za-z0-9_]*|\d+)\s+ROWS?)?',
    re.IGNORECASE,
)

_REL_HOP_RE = re.compile(r'(\[[^\]]*?\*)(\d*)\.\.(\d+)([^\]]*\])')


def _clamp_hop_bound(match: re.Match[str]) -> str:
    prefix, min_hops, max_hops, suffix = match.groups()
    if int(max_hops) > RELATE_MAX_BFS_DEPTH:
        clamped_min = str(min(int(min_hops), RELATE_MAX_BFS_DEPTH)) if min_hops else ''
        return f'{prefix}{clamped_min}..{RELATE_MAX_BFS_DEPTH}{suffix}'
    return match.group(0)


def translate_relate_query(cypher: str) -> str:
    """Translate the bounded Neo4j Cypher surface into Relate-native Cypher."""
    translated = _VECTOR_PROC_RE.sub(
        lambda m: f'SET {m.group("var")}.{m.group("prop")} = {m.group("expr")}',
        cypher,
    )
    translated = _FULLTEXT_PROC_RE.sub(r'\1)', translated)
    translated = _CALL_IN_TX_DELETE_RE.sub(
        lambda m: f'DETACH DELETE {m.group("var")}',
        translated,
    )
    translated = _REL_HOP_RE.sub(_clamp_hop_bound, translated)
    return translated
