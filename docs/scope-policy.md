# Scope Policy (Frozen)

**Status:** Frozen — Phase C / Slice 5  
**Last updated:** 2026-03

This document defines what content Bicameral ingests into the knowledge graph
by default, what is opt-in, and why.  Changes to this policy require a
deliberate PRD + changelog entry — do not drift from it silently.

---

## 1. Default Ingestion Scope: Messages Only

Bicameral's production ingestion path is **message-only** by default.

### What qualifies as a "message"

An episode with `source = EpisodeType.message` containing text authored by a
human or AI participant in a conversation.  This includes:

- User messages in a chat thread
- Assistant (AI) replies
- Narrated tool descriptions embedded in the *human turn* or *assistant turn*
  (e.g. "I ran a web search and found …")

### What is excluded by default

| Content type | `source` / role | Default? | Notes |
|-------------|----------------|----------|-------|
| Human message | `message` | ✅ **Included** | Core ingestion path |
| AI assistant message | `message` | ✅ **Included** | Core ingestion path |
| Structured JSON event | `json` | ⚙️ Opt-in | Requires explicit `source=EpisodeType.json` |
| Raw tool result blobs | `toolResult` | ❌ **Excluded** | See §2 |
| Tool call arguments | `toolCall` | ❌ **Excluded** | See §2 |
| System prompt / preamble | — | ❌ **Excluded** | High noise, low recall value |
| Embeddings / vector data | — | ❌ **Excluded** | Not human-readable facts |

### Rationale

1. **Signal / noise ratio.** Tool call arguments and raw API responses contain
   transient operational data (HTTP headers, JSON paths, cursor tokens) that
   clutter the graph without improving recall of meaningful facts.

2. **Security surface.** `toolResult` content arrives from external APIs and
   may contain adversarial text crafted to influence extraction.  Keeping it
   out of the default path reduces the injection surface.

3. **Deduplification efficiency.** Message text deduplicates cleanly by fact
   similarity.  Opaque JSON blobs resist dedup and generate spurious edges.

---

## 2. toolResult Opt-in Whitelist

Some `toolResult` types contain genuinely useful structured facts (calendar
events, contact records, financial summaries).  These may be admitted to the
knowledge graph when **all** of the following conditions are met:

1. The caller passes `source=EpisodeType.json` explicitly.
2. The tool name is on the `TOOL_RESULT_ALLOWLIST` (see below).
3. The content passes the OM compressor's strict-extraction gate
   (`extraction_mode='constrained_soft'` is recommended for these episodes).

### TOOL_RESULT_ALLOWLIST

```python
# config/tool_result_allowlist.py
# Add entries here to opt specific tool types into graph ingestion.
# Keep this list minimal; each addition expands the injection surface.

TOOL_RESULT_ALLOWLIST: frozenset[str] = frozenset({
    # Structured calendar facts (CalendarGuard-validated)
    "calendar_get_event",
    "calendar_list_events",

    # Contact/people data (confirmed safe origin)
    "contacts_lookup",

    # Restaurant/venue records (structured, human-curated)
    "restaurant_lookup",
})
```

> **To add a new entry:** open a PR that updates `TOOL_RESULT_ALLOWLIST` with
> justification, update this doc's table in §1, and add a test in
> `tests/test_tool_result_scope.py`.

---

## 3. Strict Extraction Enforcement

When processing opted-in `toolResult` or `json` episodes, callers **must** use:

```python
extraction_mode='constrained_soft'
```

This ensures:
- Edge names are canonicalized to the ontology before storage.
- Generic/noise edges (`RELATES_TO`, `MENTIONED`, etc.) are filtered.
- Punctuation-bypass variants (`RELATES^TO`, etc.) are blocked.

Using `extraction_mode='permissive'` for `toolResult` content is **not
recommended** and will emit a warning in production.

---

## 4. Contamination Prevention

Cross-lane contamination (facts from lane A appearing in lane B) violates
the memory isolation contract.  Enforce by:

1. **Always pass `group_id`** when calling `Graphiti.add_episode()`.
   Never use a shared or empty group_id in production.

2. **Run the contamination sentinel** regularly:
   ```bash
   python scripts/contamination_sentinel.py --json
   ```
   In CI: add to the scheduled nightly workflow.

3. **Edge normalization** (`scripts/normalize_edge_names.py`) ensures dedup
   doesn't silently merge facts across lanes via case-variant collisions.
   All edge names are canonicalized to SCREAMING\_SNAKE\_CASE. The normalizer
   also runs inline during extraction via `normalize_relation_type()`.

---

## 5. Episode Body Construction Rules

When constructing the body for `add_episode()`:

1. Strip metadata blocks injected by the platform (e.g. ````json {…}```
   blocks containing `message_id`, `platform`, `sender`) — use
   `_build_episode_body()` which applies this sanitization.

2. Do NOT include raw tool call JSON in the body unless the tool is on
   `TOOL_RESULT_ALLOWLIST`.

3. Prefer the human-readable narrative form over raw API response blobs.

---

## 6. Policy Change Process

| Change type | Process |
|------------|---------|
| Add to `TOOL_RESULT_ALLOWLIST` | PR + justification + test |
| Remove from allowlist | PR + migration note |
| Change default from message-only | **Full PRD required** — consult project owner |
| Change `extraction_mode` default | Full PRD required |
| Update contamination sentinel queries | PR + updated tests |

---

## 7. Quick Reference

```
Production defaults (frozen as of Phase C):
  source         = EpisodeType.message
  extraction_mode = 'permissive'  (constrained_soft for toolResult opt-ins)
  group_id       = <always required — never empty>
  toolResult     = excluded by default; opt-in via TOOL_RESULT_ALLOWLIST
```
