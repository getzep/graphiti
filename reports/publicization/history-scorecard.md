# Public History Migration Scorecard

- Generated: `2026-02-16T14:22:37Z`
- Threshold (clean fallback): `80.0`
- Rule: choose clean-foundation if filtered-history score is below threshold or an unresolved HIGH finding remains after one remediation pass.

## Weighted scores

| Candidate | Score |
| --- | --- |
| filtered-history | 39.5 |
| clean-foundation | 94.95 |

## Weights

| Metric | Weight |
| --- | --- |
| privacy_risk | 0.35 |
| simplicity | 0.35 |
| merge_conflict_risk | 0.20 |
| auditability | 0.10 |

## Candidate branches

- filtered-history: `cutover/filtered-history`
- clean-foundation: `cutover/clean-foundation`

## Decision

- Winner: `clean-foundation`
- Winner branch: `cutover/clean-foundation`
- Reason: Filtered-history has unresolved HIGH risk after one remediation pass.
- filtered unresolved HIGH: `True`

## Cutover commands (winner)

```bash
git fetch origin
git checkout --orphan cutover/clean-foundation origin/main
git add -A
git commit -m "chore(public): clean-foundation cutover snapshot"
git push --force-with-lease origin cutover/clean-foundation
# after review approval: git push --force-with-lease origin cutover/clean-foundation:main
```

## Rollback plan

```bash
git tag pre-cutover-main 7f1b83f7c5749a3f68dce147f5c29e8fdf7b2840
# if rollback is required:
git push --force-with-lease origin 7f1b83f7c5749a3f68dce147f5c29e8fdf7b2840:main
```
