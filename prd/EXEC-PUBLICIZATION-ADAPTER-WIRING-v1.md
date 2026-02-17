# PRD: Adapter Wiring (Public Template)

## Execution Status
- **Status:** Completed
- **Merged PR:** `#25`
- **Merge commit:** `0ec8853`
- **Follow-on split:** private/runtime mappings moved out of public repo in `#30` (`def709e`)

## Why this file is now a template
This PRD is intentionally retained as a **public template placeholder**.

The original execution PRD included private workflow and consumer mappings. After public/private boundary hardening, those operational mappings were moved to the private overlay repo (`graphiti-openclaw-private`) and are no longer maintained in public history.

## Public guidance
- Keep generic router framework code in this repository.
- Keep only example pack/profile mappings in this repository.
- Store real operational mappings and environment-specific runbooks in a private overlay repository.

See `docs/runbooks/runtime-pack-overlay.md` for the active split model.
