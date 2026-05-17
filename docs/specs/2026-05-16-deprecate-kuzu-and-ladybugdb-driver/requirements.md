# Requirements — deprecate Kuzu and add LadybugDB driver

## Outcome

Self-hosting users get a clear off-ramp from the archived Kuzu driver: a drop-in `LadybugDriver` ships alongside Kuzu, Kuzu becomes deprecated with a target removal version, and graph telemetry distinguishes the two backends.

## Users affected

Anyone installing `graphiti-core[kuzu]`, anyone running the `examples/quickstart` Kuzu path, the maintainers who triage Kuzu bugs (currently #1132 and #1469), and the telemetry pipeline (provider counts).

## In scope

- Add `graphiti_core/driver/ladybug/` mirroring the existing `kuzu/` shape (driver class, `record_parsers.py`, `operations/*.py`).
- Add `[project.optional-dependencies].ladybug = ["ladybugdb>=…"]` in `pyproject.toml`.
- Add `GraphProvider.LADYBUG = 'ladybug'` in `graphiti_core/driver/driver.py` and propagate to `helpers.get_default_group_id`.
- Mark Kuzu deprecated: `DeprecationWarning` in `KuzuDriver.__init__`, a "deprecated" note next to the `kuzu` extra, update the `README.md` install table and `CLAUDE.md` Database Setup section, set a removal target.
- Add `tests/driver/test_ladybug_driver.py` and `tests/driver/test_ladybug_driver_int.py` (integration, `@pytest.mark.integration`).
- Add a `quickstart_ladybugdb.py` next to `quickstart_kuzu.py` (or update the existing Kuzu quickstart to recommend Ladybug).
- Resolve GitHub issue #1132 in the PR description.

## Out of scope

- Removing the Kuzu driver in this PR. Keep one release cycle as deprecated.
- Schema migration tooling — LadybugDB is a Kuzu fork and is storage-compatible at this point per upstream.
- Any LadybugDB-specific extensions beyond parity with the current Kuzu surface.

## Decisions

- Ship LadybugDB and keep Kuzu deprecated for one release. Reason: existing users; clean off-ramp; reverting is trivial if Ladybug stalls.
- `LadybugDriver` is a standalone class, not a `KuzuDriver` subclass. Reason: lets the codebases diverge cleanly upstream without breaking our subclass invariants.
- New `GraphProvider.LADYBUG = 'ladybug'` rather than reusing `KUZU`. Reason: telemetry and diagnostics need distinct provider IDs to track migration.
- Removal target for Kuzu: graphiti-core 0.32.0 (two minor releases out). Reason: gives users two release cycles to migrate.
