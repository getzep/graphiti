"""
Copyright 2024, Zep Software, Inc.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.

Canonical-text helpers for Graphiti embedding inputs.

The embedder must receive the same text that callers use to calculate a
content hash. These helpers define that text for entity names, node content,
and edge facts.
"""

from __future__ import annotations

import unicodedata

# Keep this limit in sync with the embedder client's input cap.
MAX_INPUT_CHARS = 16_000


def canonicalize(text: str) -> str:
    """Apply NFC + truncate to MAX_INPUT_CHARS. The two steps every producer
    MUST run before either embedding or hashing."""
    nfc = unicodedata.normalize('NFC', text)
    if len(nfc) > MAX_INPUT_CHARS:
        nfc = nfc[:MAX_INPUT_CHARS]
    return nfc


def canonical_node_name_text(name: str) -> str:
    """Canonical embedder input derived from an entity node's name.

    Replaces newlines with spaces so the embedder sees a single-line input,
    matching the legacy single-item embedding path; then applies the
    universal NFC + truncation step.
    """
    return canonicalize(name.replace('\n', ' '))


# Python's str.strip() removes U+001C..U+001F. This set keeps trimming
# consistent with the Unicode whitespace characters used by the graph data.
_UNICODE_SPACE = (
    '\t\n\v\f\r \x85\xa0\u1680'
    '\u2000\u2001\u2002\u2003\u2004\u2005\u2006\u2007\u2008\u2009\u200a'
    '\u2028\u2029\u202f\u205f\u3000'
)


def canonical_node_content_text(name: str, summary: str) -> str:
    """Canonical embedder input derived from an entity node's name and summary.

    The name and summary are trimmed and joined with a blank line. When one is
    empty, the other is used alone.
    """
    name = name.strip(_UNICODE_SPACE)
    summary = summary.strip(_UNICODE_SPACE)
    if not name:
        text = summary
    elif not summary:
        text = name
    else:
        text = f'{name}\n\n{summary}'
    return canonicalize(text)


def canonical_entity_node_text(uuid: str, group_id: str, name: str, summary: str) -> str:
    """Canonical embedder input of an entity node.

    A user node (uuid == group_id) embeds only its name. Every other entity
    node embeds its name and summary.
    """
    if uuid and uuid == group_id:
        return canonical_node_name_text(name)
    return canonical_node_content_text(name, summary)


def canonical_edge_fact_text(fact: str) -> str:
    """Canonical embedder input derived from an entity edge's fact text.

    Mirrors canonical_node_name_text: collapse newlines so the embedder sees
    a single-line input, then NFC + truncate.
    """
    return canonicalize(fact.replace('\n', ' '))
