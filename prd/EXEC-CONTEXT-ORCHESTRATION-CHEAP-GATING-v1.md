# PRD: Context Orchestration v1 — Cheap Gating Fixes

## PRD Metadata
- Type: Execution
- Kanban Task: task-bicameral-context-orchestration-cheap-gating-v1
- Parent Epic: task-bicameral-context-orchestration-epic-v1
- Depends On: N/A
- Preferred Engine: Either
- Owned Paths:
  - plugin/hooks/recall.ts
  - plugin/hooks/pack-injector.ts
  - plugin/intent/detector.ts
  - plugin/hooks/capability-injector.ts
  - plugin/config.ts
  - plugin/tests/correctness.test.ts

## Overview
Reduce noisy context injection by adding deterministic per-turn gating and dedupe behavior to the bicameral OpenClaw plugin.

Currently, `<graphiti-context>` is injected on almost every turn (minPromptChars=6), even when empty. Sticky intent treats any short message as a follow-up. There is no dedupe across turns.

This execution unit adds cheap deterministic fixes — no ML classifier, no backend changes.

## Goals
1. Make null injection first-class for low-information turns (ack/yes/no/emoji/`?`/`ok`).
2. Suppress empty recall payloads (no wrapper tag when 0 facts returned).
3. Add per-turn + rolling-window dedupe/novelty filter for recalled facts.
4. Tighten sticky follow-up behavior so short acknowledgements do not keep injecting packs.
5. Keep capability injection compatibility by default (intent-gated injection enabled only when explicitly opted into strict mode).

## Definition of Done (DoD)
**DoD checklist:**
- [ ] Low-information turns (ack/yes/no/emoji/`?`/`ok`/`sounds good`/`thanks`) produce no `<graphiti-context>` block.
- [ ] Empty recall responses (0 facts) produce no injected context block at all (not even an empty wrapper).
- [ ] Duplicate fact lines within a single turn are collapsed.
- [ ] Recently injected facts (rolling 3-turn window, session-local) are suppressed unless novelty threshold is met.
- [ ] Sticky intent requires explicit follow-up signals (not just short word count).
- [ ] `capabilityRequireIntent` defaults to `false` in plugin config (strict intent-only mode is an opt-in explicit setting).
- [ ] Debug logging emits gating reasons (`skip_low_info`, `skip_empty_recall`, `deduped_N_facts`, `novelty_suppressed_N`).
- [ ] Tests cover all gating behaviors above.

**Validation commands (run from repo root):**
```bash
node --experimental-strip-types --check plugin/index.ts
node --experimental-strip-types --test plugin/tests/correctness.test.ts
```

**Pass criteria:**
- Both commands exit 0.
- New tests for low-info gate, empty suppression, dedupe, sticky hardening, and capability intent-gate all pass.

## Functional Requirements

### FR-1: Low-information turn gate
Add a deterministic classifier in `plugin/hooks/recall.ts` (before Graphiti search) that detects low-information prompts and returns early with no injection.

Patterns to match (case-insensitive, trimmed):
- Single tokens: `ok`, `yes`, `no`, `k`, `yep`, `nope`, `sure`, `thanks`, `ty`, `thx`, `cool`, `nice`, `agreed`, `done`, `noted`
- Punctuation-only: `?`, `!`, `...`, `??`, `!!`
- Emoji-only: prompt consists entirely of emoji characters
- Common short acks: `sounds good`, `got it`, `makes sense`, `lgtm`, `go ahead`, `do it`, `perfect`, `great`, `good call`

When matched, log `skip_low_info` (debug mode) and return `{ prependContext: '' }`.

### FR-2: Empty recall suppression
In `plugin/hooks/recall.ts`, after Graphiti search returns results: if `facts.length === 0` and no entities are useful, do NOT call `formatGraphitiContext`. Instead, skip the graphiti block entirely (push nothing to `parts`).

Remove the current "No relevant facts found." injection line.

### FR-3: Per-turn + rolling dedupe/novelty filter
In `plugin/hooks/recall.ts`, after receiving facts from Graphiti:

1. **Within-turn dedupe:** normalize each fact string (lowercase, collapse whitespace) and remove exact duplicates, keeping first occurrence.
2. **Rolling novelty window:** maintain a session-local ring buffer of the last 3 turns' injected fact fingerprints (sha256 of normalized text). Before injecting a fact, check if its fingerprint appears in the window. If yes, suppress it. Log `novelty_suppressed_N` with count of suppressed facts.
3. After both filters, if no facts remain, treat as empty recall (FR-2).

### FR-4: Sticky intent hardening
In `plugin/intent/detector.ts`, change `shouldStick()`:

Current behavior: if `wordCount(prompt) <= stickyMaxWords` → sticky applies.

New behavior: sticky applies ONLY when BOTH conditions are met:
1. `wordCount(prompt) <= stickyMaxWords`
2. At least one explicit follow-up signal is present in the prompt (from `stickySignals` list)

This prevents "ok" / "yes" / "?" from inheriting the previous workflow intent.

### FR-5: Capability injection intent-gate default
In `plugin/config.ts`, keep `DEFAULT_CONFIG.capabilityRequireIntent` at `false` for default compatibility.

Set it to `true` explicitly when strict intent-only capability injection is desired; this reduces noise but is opt-in for compatibility. This means capability subset injection only fires when a pack intent has been detected in strict mode.

### FR-6: Observability
All gating decisions should be logged via the existing `logger()` function (only emitted when `config.debug` is `true`):
- `skip_low_info` — low-information gate triggered
- `skip_empty_recall` — 0 facts after search
- `deduped_N_facts` — N duplicates removed within turn
- `novelty_suppressed_N` — N facts suppressed by rolling window
- `sticky_rejected_no_signal` — sticky was not applied because no follow-up signal found

## Non-Goals (Out of Scope)
- ML classifier for turn intent.
- Backend retrieval algorithm changes.
- Temporal metadata in injected context.
- Query decomposition / RLM subagents.
- Changes to pack materialization or runtime_pack_router.py.

## Technical Considerations
- Keep all gating logic deterministic (regex/string matching only).
- Rolling dedupe state is per-session (use the existing session-keyed state pattern in pack-injector).
- Do not change the GraphitiClient interface or search parameters.
- Preserve existing fallback behavior when Graphiti is unreachable.
- Ensure group chat safety is maintained (no private pack leakage from gating changes).

## Open Questions
1. Should the rolling novelty window be 3 turns or 5? (Default to 3; configurable later.)
