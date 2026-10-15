# Provenance

| Item | Value |
|---|---|
| Upstream | [getzep/graphiti](https://github.com/getzep/graphiti) (default branch `main`) |
| GitHub fork | yes (`redevops-io/graphiti`, default branch `main`) |
| Upstream commit this fork is based on (merge base) | `526dcad7a300f3c5c506ff96a68bcdc7ca9f97ed` (2026-07-09) |
| Sync status (computed 2026-10-15 via GitHub compare API) | diverged: 2 commits ahead, 138 commits behind upstream `main` |
| Licence | Apache-2.0, retained from upstream (`LICENSE`) |

## Local delta (computed with `gh api repos/getzep/graphiti/compare/main...redevops-io:graphiti:main`)

2 commits (2026-07-11): `G1: source hydration - recover raw source turns behind lossy extracted facts` and its merge commit (PR #1).
5 files, +296/-2:

- `FORK_NOTES.md` (new) - rationale and description of the change
- `graphiti_core/edges.py` (+24/-1)
- `graphiti_core/graphiti.py` (+45/-1)
- `graphiti_core/search/source_hydration.py` (new, +63)
- `tests/test_source_hydration.py` (new, +123)

Nothing else differs from upstream at the time of computation. See [FORK_NOTES.md](FORK_NOTES.md).

## Caveats

- `FORK_NOTES.md` says "`main` tracks upstream"; as of 2026-10-15 `main` is 138 commits behind upstream and contains the G1 change.
- `pyproject.toml` declares version 0.29.2 (inherited from upstream). Git tags in this fork are inherited from upstream and stop at v0.7.5; neither reflects this fork's own releases.
- The README, badges, links and repository description are upstream's. Upstream documentation does not describe the G1 change.
- This file is a point-in-time snapshot; re-run the compare to refresh.
