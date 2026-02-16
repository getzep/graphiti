# What Not to Migrate to Public

This is the explicit migration exclusion list for publicization. If unsure, do not migrate.

## Block list

- **Private workflow packs**: organization-specific playbooks, private runbooks, or personnel-facing procedure sets.
- **Content packs and drafts**: `content_strategy`, writing samples, private content drafts, and marketing narratives.
- **Private context packs**: source folders containing private context profiles and person-specific heuristics.
- **Secrets and credentials**: `.env`, tokens, local key stores, cert bundles, private signing keys.
- **Personal state artifacts**: local caches, backups, database dumps, exports with private identifiers.
- **Legacy one-off adapters**: integrations bound to private providers or private data contracts.
- **Tooling for private operators**: internal scripts and notebooks used only for one team’s workflow.

## Decision rule

If an artifact is required by this public repository because it changes core behavior for everyone, keep it here.
If it exists to support one private workflow, keep it outside public release scope.

## Suggested flow

1. Capture private intent as a public abstraction request.
2. Implement the abstraction as a stable interface or adapter contract.
3. Keep concrete implementations out of the public foundation until generalized.
