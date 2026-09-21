# Issue / PR intake automation

A fully deterministic triage bot for issues and pull requests. `decide.py`
fetches the item through the GitHub API and computes a decision from objective
facts only — the issue-form sections, the labels the forms apply, the changed
file paths, and the `Fixes #<number>` references in the body. `apply.py` turns
that decision into allowlisted labels and one templated "sticky" comment. No
LLM is involved and no API key secret is needed: both scripts run on the
standard `GITHUB_TOKEN` and the Python standard library alone.

## One-time setup

1. **Create the labels** (required before the issue forms or bot run — GitHub
   silently drops form labels that do not exist):

   ```bash
   bash .github/scripts/setup-triage-labels.sh
   ```

2. **(Optional) repository variable** to disable intake without editing code:

   | Variable | Default | Purpose |
   |----------|---------|---------|
   | `INTAKE_ENABLED` | unset (enabled) | Set to `false` to disable all intake + stale workflows (kill switch). |

## How it runs

- **`.github/workflows/issue-intake.yml`** — on issue open/edit/reopen. One
  job: `decide.py --kind issue` then `apply.py --write`.
- **`.github/workflows/pr-intake.yml`** — on PR open/sync/reopen/edit, for
  same-repo and fork branches. It uses `pull_request_target` so the token can
  label fork PRs, checks out the base-branch scripts (never the PR's), and
  reads the PR through the API, so it never runs PR-authored code.
- **`.github/workflows/ai-moderator.yml`** — spam and AI-generated-content
  detection for new issues and comments (the `github/ai-moderator` action).
- **`.github/workflows/stale.yml`** — daily; warns then closes items that keep
  any `needs-*` label (`needs-info`, `needs-issue`, `needs-rfc`, `needs-tests`,
  `needs-rework`) for 14 days. `rfc-approved` and `security` items are exempt.

Trusted authors (`OWNER`/`MEMBER` author association) and draft pull requests
are skipped: `decide.py` writes the empty decision `{}`, which `apply.py`
treats as a clean no-op.

## Issue rules

The issue forms (`.github/ISSUE_TEMPLATE/*.yml`) render into `### <Label>`
sections in the body. A section counts as missing when it is absent, blank, or
`_No response_`.

- **`category`**: the first of `bug`, `feature`, `documentation`, `question`
  already on the issue (the form sets it), else `other`.
- **`areas`**: from the `Affected component` (or `Component`) dropdown —
  `graphiti-core` → `scope:core`, `MCP server` → `scope:mcp`, `REST server` →
  `scope:service`, `Documentation or examples` → `scope:docs`,
  `CI, Docker, or release` → `scope:ci`. Other values and a missing section
  give no scope; a `documentation` issue with no scope defaults to
  `scope:docs`.
- **Required sections** (missing ones produce `needs-info` and land in
  `missing_fields`):

  | Category | Required |
  |----------|----------|
  | bug | description, reproduction, expected, actual, environment |
  | feature | problem, outcome |
  | question | goal, attempted, environment |
  | documentation | location, problem |

- **`needs-rfc`**: a feature whose `Size` is `Large feature requiring design
  approval` gets `needs-rfc` unless the issue already has `rfc-approved`, and
  `proposal`/`alternatives`/`impact` become required as well.
- **`comment_id`**: `ask_rfc_fields` when `needs-rfc`; `ask_repro` for a bug
  missing fields; `ask_info` for anything else missing fields; else none.

## Pull request rules

- **`areas`**: derived from the changed file paths (see `PATH_SCOPE_RULES`).
- **`needs-issue`**: no linked issue (`Fixes #<number>`) → `needs-issue` and
  the `linked-issue` missing field. A number that 404s or resolves to a pull
  request does not count.
- **`category`**: `feature`/`bug`/`documentation` when a linked issue carries
  that label, else `feature` when the template's Feature checkbox is ticked,
  else `other`.
- **`needs-rfc`**: a feature PR whose linked issue lacks `rfc-approved`.
- **`needs-tests`**: the diff touches code but no test file, and the
  `Tests are not applicable` checkbox is not ticked.
- **`needs-rework`**: `needs-issue` and `needs-tests` together.
- **`comment_id`**: `pr_needs_rework` > `pr_needs_rfc` > `pr_needs_tests`.

## Files

| File | Role |
|------|------|
| `decide.py` | Read-only: fetch item, compute the deterministic decision, write JSON. Stdlib only. |
| `apply.py` | Deterministic: validate the decision, apply allowlisted labels + one templated sticky comment. Stdlib only. |
| `decision.schema.json` | The decision contract and the **single source of truth** for the label taxonomy (apply.py derives its allowlist from it). |
| `templates/*.md` | The only text the bot can post; substitutions come from closed sets. |

Tests: `tests/intake/` (run by `unit_tests.yml`). The workflow guard tests in
`test_workflows.py` encode the security invariants — keep them passing.
