# Issue / PR intake automation

A classify-then-apply bot that triages issues and pull requests. A read-only
`classify` step asks an LLM for a structured decision; a separate write-scoped
`apply` step turns that decision into allowlisted labels and one templated
"sticky" comment. The model has no tools and no write token, and `apply.py`
re-validates every decision, so a prompt-injected model can at worst produce a
wrong-but-valid decision — never an arbitrary write.

## One-time setup

1. **Create the labels** (required before the issue forms or bot run — GitHub
   silently drops form labels that do not exist):

   ```bash
   bash .github/scripts/setup-triage-labels.sh
   ```

2. **Add the LLM API key** as a repository secret:

   - `INTAKE_API_KEY` — a key for your OpenAI-API-compatible endpoint.

3. **(Optional) repository variables** to tune behavior without editing code:

   | Variable | Default | Purpose |
   |----------|---------|---------|
   | `INTAKE_MODEL` | `gpt-4.1-mini` | Model id for the endpoint. |
   | `INTAKE_BASE_URL` | unset (OpenAI SDK default) | Point at any OpenAI-API-compatible endpoint. |
   | `INTAKE_ENABLED` | unset (enabled) | Set to `false` to disable all intake + stale workflows (kill switch). |

## How it runs

- **`.github/workflows/issue-intake.yml`** — on issue open/edit/reopen, and on
  new comments while an issue carries `needs-info`.
- **`.github/workflows/pr-intake.yml`** — on PR open/sync/reopen/edit, for
  same-repo branches only. It checks out the base-branch scripts (not the PR's),
  so it never runs PR-authored code with the token.
- **`.github/workflows/stale.yml`** — daily; warns then closes items that keep
  any `needs-*` label (`needs-info`, `needs-issue`, `needs-rfc`, `needs-tests`,
  `needs-rework`) for 14 days. `rfc-approved` and `security` items are exempt.

Provider/model swap = change `INTAKE_MODEL` / `INTAKE_BASE_URL`. Because
`apply.py` is the validation gate, structured-output strictness varies by model
without weakening safety — but verify a new model returns schema-valid JSON.

## What is deterministic vs. model-decided

Objective facts are computed in code, not left to the model:

- **PR scope** (`scope:*`) is derived from the changed file paths.
- **`needs-issue`** is applied to any PR with no linked issue.

The model only makes the genuinely fuzzy calls (category, duplicate, whether a
repro is present, whether a large change needs an RFC).

## Files

| File | Role |
|------|------|
| `classify.py` | Read-only: fetch item, redact secrets, call the LLM, write a decision. Needs `requirements.txt` (openai). |
| `apply.py` | Deterministic: validate the decision, apply allowlisted labels + one templated sticky comment. Stdlib only. |
| `decision.schema.json` | The decision contract and the **single source of truth** for the label taxonomy (apply.py derives its allowlist from it). |
| `prompts/*.md` | Trusted classification instructions (sent as the system role). |
| `templates/*.md` | The only text the bot can post; substitutions come from closed sets. |

Tests: `tests/intake/` (run by `unit_tests.yml`). The workflow guard tests in
`test_workflows.py` encode the security invariants — keep them passing.
