# Backup Readiness Report Template

This public repository intentionally ships a **template** only.

Real readiness reports are environment-specific operational artifacts and should be tracked in a private overlay repository (for example: `graphiti-openclaw-private`).

## Suggested private report fields
- run timestamp (with timezone)
- operator / runner identity
- command transcript hashes
- backup snapshot manifest path
- restore-test result
- GO / NO-GO decision and blocking conditions
- rollback decision notes

## Public policy
- Keep this file as a template.
- Do not commit real operational readiness evidence or private runtime paths here.
