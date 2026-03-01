# Local Hotfix Registry (Graphiti Core)

This repository (`bicameral`) maintains intentional local behavioral deviations from upstream `graphiti_core` inside the `patches/` directory.

Because the velocity of upstream `getzep/graphiti` is high, we track explicit patch files rather than relying solely on git rebase/merge history to survive conflict resolution.

## Active hotfixes (Graphiti Core)

### 1) Deterministic Migration Dedupe Mode
- Purpose: Prevent semantic duplicate resolution instability (`invalid duplicate_facts idx`) during bulk backfill by disabling semantic (LLM-based) re-evaluation for exact node matches.
- Files: `graphiti_core/graphiti.py`
- Rationale: Mandatory for Gate 3 reliable curation loading; migration-only parameter thread.

### 2) Malformed `RELATES_TO` Edge Guarding
- Purpose: Prevent `ValidationError` hydration failures downstream when legacy edges miss `uuid`, `group_id`, or `episodes`.
- Files: `graphiti_core/edges.py`, `graphiti_core/utils/maintenance/edge_operations.py`
- Rationale: Schema divergence existed dynamically in legacy data; defensive fallback required inside the core hydration paths.

### 3) Trust-Aware Retrieval & Ranking Additions
- Purpose: Allow overlay scores/thresholds to boost canonical facts (`ingest_curated_facts` vs LLM extractions).
- Files:
  - `graphiti_core/search/search.py`
  - `graphiti_core/search/search_config.py`
  - `graphiti_core/search/search_config_recipes.py`
  - `graphiti_core/search/search_utils.py`
- Rationale: Essential behavior layer for `bicameral` trust topology; cannot be pushed to runtime solely via hooks due to hardcoded score aggregation in `graphiti_core/search`. Will be re-evaluated as upstream search APIs mature.

### 4) Constrained-Soft Extraction Mode
- Purpose: Two-mode extraction architecture for ontology-aware graph lanes.
  - **permissive** (default): existing broad extraction, unchanged.
  - **constrained_soft**: ontology-conformant mode using dedicated prompt branches
    (not appended to permissive prompt — avoids conflicting directives) plus
    code-level enforcement after LLM extraction:
    - `_normalize_relation_type()`: normalize LLM relation type output to
      SCREAMING_SNAKE_CASE (trim whitespace, spaces/hyphens → underscores, uppercase)
      before any ontology comparison — catches mixed-case/spaced LLM variants.
    - `_canonicalize_edge_name()`: snap near-miss relation types to ontology names
      using difflib SequenceMatcher (threshold ≥ 0.78) on normalized form.
      Negation polarity guard prevents canonicalization from flipping NOT_* to
      non-NOT_* names (and vice versa) to avoid semantic inversions.
    - `_should_filter_constrained_edge()`: drop generic off-ontology edges
      (RELATES_TO, MENTIONS, IS_RELATED_TO, etc.) using case-insensitive
      normalized comparison.
    - Node strictness: in constrained_soft mode with custom ontology types present,
      entities whose type resolves to the generic Entity fallback (type_id=0 or
      invalid) are dropped post-extraction. Logged with dropped count.
- Files:
  - `graphiti_core/prompts/extract_edges.py` — mode-specific prompt dispatch (`_edge_permissive`, `_edge_constrained_soft`); LANE_INTENT references marked "if provided"
  - `graphiti_core/prompts/extract_nodes.py` — mode-specific prompt dispatch for all episode types; LANE_INTENT references marked "if provided"
  - `graphiti_core/utils/maintenance/edge_operations.py` — enforcement helpers + extraction_mode param
  - `graphiti_core/utils/maintenance/node_operations.py` — extraction_mode param thread + node strictness filter
  - `graphiti_core/graphiti.py` — extraction_mode param in add_episode + _extract_and_resolve_edges
- Rationale: Permissive extraction creates noise in ontology lanes (RELATES_TO edges, near-miss
  edge names). Code enforcement is more reliable than prompt-only guidance, which is subject to
  LLM drift. Intent guidance comes from per-lane YAML config (intent_guidance / extraction_emphasis).
- Backward compatibility: extraction_mode defaults to 'permissive' at all call sites. No
  existing behavior changed. Permissive lanes are completely unaffected.
- Added to allowlist: `graphiti_core/prompts/extract_edges.py`,
  `graphiti_core/prompts/extract_nodes.py`, `graphiti_core/utils/maintenance/node_operations.py`
- Patches: `patches/graphiti_core/prompts/*.patch`, `patches/graphiti_core/utils/maintenance/*.patch`,
  `patches/graphiti_core/graphiti.py.patch`

### 5) Phase 3C Craft Candidate Filtering + Dropped-Candidate Logging
- Purpose: Harden the three Phase 3C craft lanes by filtering low-signal/meta extraction candidates before graph write and emitting structured dropped-candidate logs for quality-gate diagnostics.
- Files:
  - `graphiti_core/graphiti.py`
- Behavior:
  - Scope-limited to `s1_writing_samples`, `s1_inspiration_short_form`, `s1_inspiration_long_form`
  - Pre-resolution filter removes clearly meta/tautological node candidates
  - Pre-write filter enforces required craft fields (`evidence_span`, `craft_type`, `pattern_template`, `when_to_use`) for non-anchor entities, evidence support checks against episode text, and drops meta/tautological facts
  - Writes JSONL drop logs (default under `reports/canonical-truth/`) with reject reasons and source metadata
- Rationale: Private overlay-only changes cannot persist core extraction behavior because runtime rebuild hard-resets to public `origin/main` before reapplying overlay. This behavior must live in public core to survive deterministic rebuild and unblock 3C extraction quality gates.
- Notes: Log output path can be overridden with `GRAPHITI_DROPPED_CANDIDATES_LOG`.

## How to Sync Upstream

To safely absorb upstream updates while keeping these hotfixes:

1. **Start Sync Branch:** Create a branch `sync/upstream-YYYYMMDD` from `origin/main`.
2. **Merge Upstream:** `git pull upstream main`
3. **Resolve Conflicts (Upstream Wins in core):** If conflicts arise in `graphiti_core/**`, accept upstream's version. You do not manually rebuild the logic during conflict resolution.
4. **Re-Apply Patches:** Run `git apply patches/graphiti_core/*` to neatly apply our explicit hotfixes over the fresh upstream baseline.
5. **Re-Export Patches:** If upstream structural changes occurred (e.g. they moved code blocks around), run `./scripts/export-core-patches.sh` post-validation to update the line numbers in the stored patch files for the next release.
6. **Guard Check:** The CI run will verify no other files in `graphiti_core/**` were modified besides what is listed in `config/graphiti_core_allowlist.txt`.
