# Dual-Brain Operators Guide

This runbook explains how to operate the Dual-Brain system in practice: when to approve facts, how to handle conflicts, and how to debug when Brain 1 and Brain 2 disagree.

---

## Quick Reference: The Two Brains

| Aspect | Brain 1 (Neo4j) | Brain 2 (Ledger) |
|--------|----------------|-----------------|
| **Type** | Semantic graph | Append-only ledger |
| **Source of truth?** | No. Derived. | **Yes. Canonical.** |
| **Update speed** | Fast (extraction runs continuously) | Slow (approval is gated) |
| **Audit trail** | Partial (timestamps, extraction metadata) | **Complete (full provenance)** |
| **Rollback?** | Messy (requires re-extraction or manual edge deletion) | **Clean (single ledger entry)** |
| **Trust decisions** | None (all edges equal until synced) | **Yes (promotion policy)** |

## Daily Operations

### Monitoring the Candidates Queue

Brain 2 starts as a quarantine zone. New extractions land in `candidates.db` waiting for promotion.

```bash
# Count pending candidates by type
python3 scripts/monitor_candidates.py --summary

# Show candidates pending human review (flagged for conflicts)
python3 scripts/monitor_candidates.py --requires-approval

# Show candidates auto-promoted today
python3 scripts/monitor_candidates.py --auto-promoted --since 1d
```

### Approving or Rejecting Candidates

When a candidate requires approval, review the source evidence and decide:

```bash
# Inspect a specific candidate
python3 scripts/review_candidate.py --id <candidate-uuid>

# Approve it (promotes to Brain 2 Ledger)
python3 scripts/review_candidate.py --id <candidate-uuid> --approve

# Reject it (stays quarantined, or marked as invalid)
python3 scripts/review_candidate.py --id <candidate-uuid> --reject --reason "contradicts my actual preference"
```

### Handling Conflicts

When a new fact contradicts an existing promoted fact, the system flags it automatically:

```bash
# Show pending conflicts
python3 scripts/review_candidate.py --conflicts-only

# Resolve a conflict
python3 scripts/review_candidate.py --id <new-uuid> --supersede <old-uuid> --reason "switched from coffee to matcha"
```

When you supersede a fact, Brain 2 records it:
- Old fact marked as `superseded_by: <new-uuid>`
- New fact gets `supersedes: <old-uuid>`
- Timestamp recorded in ledger hash chain

Brain 1 (Neo4j) doesn't get updated immediately. The next sync run will set `trust_score = 0` on the old edge and `trust_score = 1.0` on the new one.

### Syncing Brain 2 Changes Back to Brain 1

After approvals, you must sync to update Neo4j with new trust scores:

```bash
# Dry-run: show what will change
python3 scripts/sync_trust_scores.py --backend neo4j --dry-run

# Execute the sync
python3 scripts/sync_trust_scores.py --backend neo4j --execute

# Verify results
python3 scripts/sync_trust_scores.py --backend neo4j --verify
```

The sync script:
1. Reads all promoted facts from `state/fact_ledger.db`
2. Matches them to Neo4j edges via UUID
3. Sets `trust_score = 1.0` on promoted edges
4. Sets `trust_score = 0.25` on corroborated (non-promoted) edges
5. Aggregates entity-level trust scores (max of connected edges)

---

## Debugging Disagreements

### "Brain 1 says X, Brain 2 says Y"

This happens when the two systems are out of sync. Diagnose with:

```bash
# Show facts in Brain 2 that don't have matching edges in Brain 1
python3 scripts/diagnose_brain_sync.py --missing-in-neo4j

# Show edges in Brain 1 that don't have a Brain 2 record
python3 scripts/diagnose_brain_sync.py --orphaned-in-neo4j

# Show mismatched trust scores
python3 scripts/diagnose_brain_sync.py --trust-score-mismatch
```

Common causes:
- **Sync script hasn't run since approval.** Run `sync_trust_scores.py --execute`.
- **Graph was re-extracted (FalkorDB → Neo4j)** and UUID mappings are stale. Re-link candidates via `scripts/relink_candidates_post_migration.py`.
- **Manual Neo4j edit** bypassed Brain 2. Audit the change: `git log state/fact_ledger.db`. If it was wrong, revert with `scripts/rollback_ledger.py --to <commit-hash>`.

### "The LLM is over-confident about a fact"

Brain 1's confidence is not a truth signal. It's just how often the LLM saw the same extraction pattern.

If the LLM is repeatedly extracting something wrong:

1. **Reject it in Brain 2** to prevent auto-promotion: `review_candidate.py --id <uuid> --reject`
2. **Diagnose the extraction rule** causing the false positive: `scripts/audit_extraction_ontology.py --predicate "<predicate-name>"`
3. **Update the ontology** (if needed) to be more specific: edit `config/extraction_ontologies.yaml`
4. **Re-extract** the offending sessions to test the new rule: `scripts/re_extract_sessions.py --since 7d --dry-run`

### "I approved a fact in Brain 2, but it's not ranking higher in Brain 1 retrieval"

Possible causes:
1. **Sync hasn't run.** Check: `python3 scripts/sync_trust_scores.py --verify`
2. **Trust weight is set to 0.** Check env var: `echo $GRAPHITI_TRUST_WEIGHT`
3. **The UUID link is broken.** After graph migrations, candidates may have stale edge UUIDs. See "Graph was re-extracted" above.
4. **Semantic relevance is too low.** Even with `trust_score = 1.0`, if the fact is semantically irrelevant to the query, it won't rank high. This is correct behavior. Trust boosts, not transforms, relevance.

---

## Operational Policies

### Auto-Promotion Rules

Facts auto-promote (no human review needed) if:
- **Author:** Fact is from your own authorship (not from a group chat or external source)
- **Confidence:** LLM confidence ≥ 0.90
- **Risk tier:** Low-risk (preferences, style, non-critical identity)
- **Conflict:** No conflict with existing promoted facts, OR the new fact is newer and more reliable

Examples that auto-promote:
- "I prefer meetings after 10am" (you stated it directly, low risk)
- "I switched from coffee to matcha" (you stated it, supersedes old preference)

Examples that require approval:
- "Yuan is CEO of TechCorp" (too risky without verification)
- "I'm quitting my job" (high risk, only auto-promote if explicitly restated 2+ times)
- "My family thinks I should X" (third-party claim, stays quarantined)

Update auto-promotion rules in `config/promotion_policy.yaml`.

### Candidates That Never Auto-Promote

These stay quarantined unless you explicitly approve:
- Facts from group chats or untrusted sources
- Claims about other people (even if from your transcript)
- Hypotheticals and questions ("should I move to SF?" ≠ "I'm moving to SF")
- Contradictory facts (until you resolve which is true)

---

## Ledger Auditing

### Checking Provenance of a Fact

```bash
# Show the full history of a fact and all its versions
python3 scripts/audit_fact_provenance.py --fact-id <fact-uuid>

# Output includes:
# - Original extraction (session, timestamp, source text)
# - Promotion decision (who approved, when, reasoning)
# - All supersessions and invalidations
# - Current status and trust score
```

### Reverting a Mistaken Promotion

```bash
# Revert a specific fact to its prior state
python3 scripts/rollback_ledger.py --fact-id <uuid> --reason "promoted in error"

# Revert all changes since a given time
python3 scripts/rollback_ledger.py --since "2026-02-20T15:00Z" --reason "bad extraction run"

# Revert and audit
python3 scripts/rollback_ledger.py --fact-id <uuid> --dry-run  # see what changes
```

Rollbacks are ledger entries themselves, fully auditable.

### Ledger Hash Chain Integrity Check

```bash
# Verify the hash chain is unbroken
python3 scripts/verify_ledger_integrity.py

# Check for gaps or corruption
python3 scripts/verify_ledger_integrity.py --verbose
```

---

## Migration: Starting with Existing Neo4j Data

If you have an existing Neo4j graph and want to bootstrap Brain 2 from it:

```bash
# Initialize ledger from current graph state
python3 scripts/bootstrap_ledger_from_neo4j.py --graph-id s1_sessions --create-candidates

# This:
# 1. Exports all RELATES_TO edges from Neo4j
# 2. Creates initial candidates.db entries
# 3. Calculates initial confidence from extraction metadata
# 4. Sets all high-confidence facts to auto-promoted status
# 5. Requires your explicit review before writing to ledger

# Review the bootstrap set
python3 scripts/review_candidate.py --bootstrap-only --summary

# Approve the bootstrap
python3 scripts/review_candidate.py --bootstrap-only --approve-all
```

This is a one-time operation. After bootstrap, Brain 2 is the source of truth.

---

## Troubleshooting

### "Sync is slow"

If `sync_trust_scores.py` takes >5 minutes:

```bash
# Check Neo4j performance
python3 scripts/diagnose_brain_sync.py --neo4j-perf

# May be:
# - Graph is huge (>100k edges). This is normal. Scale Neo4j heap if needed.
# - UUID matching is failing (re-link candidates if graph was migrated).
# - Batch size is too small. Increase `SYNC_BATCH_SIZE` env var (default 1000).
```

### "Candidates.db is bloated"

If `state/candidates.db` is >1GB:

```bash
# Compact and analyze
python3 scripts/maintain_candidates_db.py --compact

# Show age distribution
python3 scripts/maintain_candidates_db.py --age-stats

# Archive old rejected candidates (older than 90 days, status=rejected)
python3 scripts/maintain_candidates_db.py --archive-rejected --before 90d
```

### "Trust scores aren't applying"

```bash
# Diagnose
python3 scripts/diagnose_brain_sync.py --trust-score-mismatch

# Check the trust_weight setting
echo "Current trust_weight: $GRAPHITI_TRUST_WEIGHT"

# If set to 0, trust boosting is disabled
# To enable: export GRAPHITI_TRUST_WEIGHT=0.15

# Re-sync
python3 scripts/sync_trust_scores.py --execute
```

---

## See Also

- [Promotion Policy](../promotion-policy.md) — Rule specification for what gets auto-promoted
- [Fact Ledger Schema](../fact-ledger-schema.md) — SQLite structure and hash-chain design
- [The Dual-Brain Architecture](../DUAL-BRAIN-ARCHITECTURE.md) — Conceptual overview
