# Validation — deprecate Kuzu and add LadybugDB driver

## Automated tests

- `tests/driver/test_ladybug_driver.py` — mock-level coverage of save / get / delete for `EntityNode`, `EntityEdge`, `EpisodicNode`, `EpisodicEdge`, `CommunityNode`, `CommunityEdge`, `SagaNode`, `HasEpisodeEdge`, `NextEpisodeEdge`. Mirrors `tests/driver/test_falkordb_driver.py` shape.
- `tests/driver/test_ladybug_driver_int.py` — `@pytest.mark.integration` end-to-end ingest + search against a live LadybugDB instance.
- `tests/test_graphiti_mock.py` — add a parametrize entry so the existing mock suite also covers the new driver where applicable.
- `tests/test_kuzu_deprecation.py` — asserts a `DeprecationWarning` fires on `KuzuDriver()` instantiation and the warning text mentions both `LadybugDriver` and the 0.32.0 removal target.

## Smoke checks

```bash
uv pip install -e .[ladybug]
uv run python -c "from graphiti_core.driver.ladybug.ladybug_driver import LadybugDriver; print('import ok')"
uv run python examples/quickstart/quickstart_ladybugdb.py
```

A successful run prints extracted nodes, edges, and search results identical in shape to the Kuzu quickstart.

## Manual criteria

- The `DeprecationWarning` from `KuzuDriver.__init__` is visible in `pytest -W default` output but doesn't spam every line of CI.
- README install table is unambiguous: Ladybug is the recommended path; Kuzu is marked deprecated with the removal version.

## Risks & rollback

- **Failure modes**: LadybugDB diverges from Kuzu's wire/storage layout sooner than expected (forces real porting work); the LadybugDB fork itself goes unmaintained; integration tests are flakey because the public LadybugDB image is missing or unstable; users on Kuzu storage files can't open them with Ladybug.
- **Rollback**: revert the PR. Kuzu remains usable (just archived). No data migration is involved, so no destructive state.

## Open questions

- Removal version for Kuzu: 0.32.0 is in the decisions section, but if Ladybug isn't stable enough by then we may need to extend. Re-evaluate before tagging 0.31.
- Will LadybugDB ship an official Docker image we can wire into `docker-compose.test.yml`? If not, keep the integration test gated behind `RUN_LADYBUG_INT=1` and skip from the default CI matrix.
- Does Kuzu issue #1469 (Windows access violation) get backported one last time before deprecation, or is it parked immediately? Defer to maintainer judgment.
