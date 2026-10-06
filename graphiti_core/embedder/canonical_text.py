"""Canonical-text helpers for embedding inputs.

Producers in the wider Zep system hash the canonical text and pass it to the
graph-service durable embedding store under that hash (spec-20 §2.3, §2.8).
The hash must be over the same bytes the embedder receives, which means the
text-derivation, Unicode normalization, and length cap must all happen in
exactly one place. These helpers are that place for graphiti's entity-node
names and entity-edge facts.

The canonicalization steps applied here (matching spec-20 §2.8):
  1. Producer-specific text shaping (newlines collapsed to spaces, to match
     the legacy embedding path).
  2. Unicode NFC normalization so that NFD/NFC inputs hash and embed
     identically.
  3. Right-truncation to MAX_INPUT_CHARS, matching the embedder client's
     safety-net cap. Without this, a >MAX_INPUT_CHARS input gets embedded as
     a prefix while the hash spans the full string, breaking
     "exact bytes hashed match exact bytes embedded."

If you change any of these transformations, bump the canonicalization-version
segment (`c`) of the producer's `embedder_model_id` so old and new vectors do
not collide on a content-hash key.
"""

from __future__ import annotations

import unicodedata

# Must stay in sync with the embedder client's input cap. Both Gemma embedder
# clients (Python: facts_service.gemma_embedder_grpc.MAX_INPUT_LENGTH;
# Go: lib/embedder/client.go:maxInputChars) truncate to 16,000 codepoints.
# Producers apply this cap *before* hashing so the byte sequence handed to
# the embedder is exactly the byte sequence hashed.
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


def canonical_edge_fact_text(fact: str) -> str:
    """Canonical embedder input derived from an entity edge's fact text.

    Mirrors canonical_node_name_text: collapse newlines so the embedder sees
    a single-line input, then NFC + truncate.
    """
    return canonicalize(fact.replace('\n', ' '))
