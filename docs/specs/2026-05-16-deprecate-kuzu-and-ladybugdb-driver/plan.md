# Plan — deprecate Kuzu and add LadybugDB driver

## Approach

Add a new `graphiti_core/driver/ladybug/` package whose initial implementation forks the corresponding `driver/kuzu/` files (Ladybug is API-compatible at the moment). Wire it into the `GraphProvider` enum, `pyproject.toml` extras, examples, telemetry, and tests. Mark Kuzu deprecated in code, docs, and the install table with a target removal version. Keep the Kuzu driver functional through the deprecation window so no user is forced to migrate inside a single release.

## Steps

1. Add the extra to `pyproject.toml`: `ladybug = ["ladybugdb>=<latest>"]`. Add `LadybugDB` to the dev extra bundle. Bump version in `[project]`.
2. Add `GraphProvider.LADYBUG = 'ladybug'` in `graphiti_core/driver/driver.py`. Update `helpers.get_default_group_id` to handle the new provider.
3. Create `graphiti_core/driver/ladybug/` with `__init__.py`, `ladybug_driver.py`, `record_parsers.py`, and `operations/{entity_edge_ops,entity_node_ops,episode_node_ops,episodic_edge_ops,community_edge_ops,community_node_ops,saga_node_ops,has_episode_edge_ops,next_episode_edge_ops,graph_ops,search_ops}.py` — initial pass copies the Kuzu equivalents and swaps the import.
4. Re-export `LadybugDriver` from `graphiti_core/driver/__init__.py`.
5. Update telemetry: `graphiti_core/telemetry/telemetry.py` already reads the provider id; nothing to change beyond confirming `'ladybug'` flows through.
6. Add `tests/driver/test_ladybug_driver.py` (mock) and `tests/driver/test_ladybug_driver_int.py` (integration). Add LadybugDB env vars to `docker-compose.test.yml` if/when a container image is published; otherwise gate the integration test behind `RUN_LADYBUG_INT=1`.
7. Add `examples/quickstart/quickstart_ladybugdb.py` (mirror of `quickstart_kuzu.py`).
8. Mark Kuzu deprecation: emit `DeprecationWarning('KuzuDriver is deprecated; switch to LadybugDriver. Removal in graphiti-core 0.32.0')` from `KuzuDriver.__init__`. Add a `> [!WARNING]` block to the README's Kuzu section. Update the install table. Update `CLAUDE.md` Database Setup notes.
9. Update `docs/specs/roadmap.md`: move "Kuzu on Windows (#1469)" into Parked with reason "Kuzu archived upstream; tracked via LadybugDB"; add LadybugDB driver to Now.
10. Resolve #1132 in PR description.

## Dependencies / order

Step 1 before 3 (so the import works). Step 2 before 3. Steps 6–8 can land in parallel after 3–4. Step 9 last.
