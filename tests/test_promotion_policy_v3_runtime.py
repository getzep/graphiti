from __future__ import annotations

import importlib
from dataclasses import dataclass
from types import SimpleNamespace
from typing import Any

import pytest

promotion_policy_v3 = importlib.import_module("truth.promotion_policy_v3")


class _FakeResult:
    def __init__(self, row: dict[str, Any] | None, nodes_created: int = 0) -> None:
        self._row = row
        self._summary = SimpleNamespace(counters=SimpleNamespace(nodes_created=nodes_created))

    def single(self) -> dict[str, Any] | None:
        return self._row

    def consume(self) -> Any:
        return self._summary


class _FakeSession:
    def __init__(self, *, nodes_created: int) -> None:
        self.nodes_created = nodes_created
        self.calls: list[tuple[str, dict[str, Any]]] = []

    def __enter__(self) -> _FakeSession:
        return self

    def __exit__(self, exc_type, exc, tb) -> None:  # type: ignore[no-untyped-def]
        return None

    def run(self, query: str, params: dict[str, Any]) -> _FakeResult:
        self.calls.append((query, params))
        if "MERGE (c:CoreMemory" in query:
            return _FakeResult(
                {
                    "core_memory_id": params["core_memory_id"],
                    "promoted_at": params["promoted_at"],
                    "candidate_id": params["candidate_id"],
                },
                nodes_created=self.nodes_created,
            )
        return _FakeResult({"rel_count": 1})


class _FakeDriver:
    def __init__(self, *, nodes_created: int) -> None:
        self.nodes_created = nodes_created
        self.last_session: _FakeSession | None = None

    def session(self, database: str | None = None) -> _FakeSession:
        self.last_session = _FakeSession(nodes_created=self.nodes_created)
        return self.last_session


@dataclass
class _ClosableSentinel:
    closed: bool = False

    def close(self) -> None:
        self.closed = True


def _verification(candidate_id: str = "cand-1") -> Any:
    return promotion_policy_v3.VerificationRecord(
        candidate_id=candidate_id,
        verification_status="corroborated",
        evidence_source_ids=["m1", "m2"],
        verifier_version="vtest",
        verified_at="2026-01-01T00:00:00Z",
    )


@pytest.mark.parametrize("nodes_created,expected_created", [(1, True), (0, False)])
def test_promote_candidate_created_is_deterministic_from_merge_counters(
    nodes_created: int,
    expected_created: bool,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Runtime-only write-path test: disable v3 gate so candidates DB is not required.
    monkeypatch.setenv("GRAPHITI_POLICY_V3_ENABLED", "0")
    driver = _FakeDriver(nodes_created=nodes_created)

    result = promotion_policy_v3.promote_candidate(
        candidate_id="cand-1",
        verification=_verification("cand-1"),
        hard_block_check=lambda _: False,
        neo4j_driver=driver,
    )

    assert result["promoted"] is True
    assert result["created"] is expected_created
    assert result["supports_core_edges_attempted"] == 2


def test_promote_candidate_omnode_not_found_skips_gracefully(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """promote_candidate must return promoted=False/reason=omnode_not_found (not raise)
    when the OMNode is absent from Neo4j.  Covers verification-only promotion
    paths where the verification record exists in candidates.db but the source
    OMNode has been evicted or was never written to the graph.
    """

    # Runtime-only write-path test: disable v3 gate so candidates DB is not required.
    monkeypatch.setenv("GRAPHITI_POLICY_V3_ENABLED", "0")

    class _SessionNoOMNode(_FakeSession):
        def __init__(self) -> None:
            super().__init__(nodes_created=0)

        def run(self, query: str, params: dict[str, Any]) -> _FakeResult:
            self.calls.append((query, params))
            if "MERGE (c:CoreMemory" in query:
                # Simulate OMNode absent: MATCH finds no row, single() → None
                return _FakeResult(None, nodes_created=0)
            return _FakeResult({"rel_count": 1})

    class _DriverNoOMNode(_FakeDriver):
        def __init__(self) -> None:
            super().__init__(nodes_created=0)

        def session(self, database: str | None = None) -> _SessionNoOMNode:
            self.last_session = _SessionNoOMNode()
            return self.last_session

    result = promotion_policy_v3.promote_candidate(
        candidate_id="ghost-cand-1",
        verification=_verification("ghost-cand-1"),
        hard_block_check=lambda _: False,
        neo4j_driver=_DriverNoOMNode(),
    )

    assert result["promoted"] is False
    assert result["reason"] == "omnode_not_found"
    assert result["candidate_id"] == "ghost-cand-1"
    assert "core_memory_id" in result


def test_shared_driver_is_lazy_singleton(monkeypatch: pytest.MonkeyPatch) -> None:
    sentinel = _ClosableSentinel()
    load_count = 0

    def _fake_loader() -> _ClosableSentinel:
        nonlocal load_count
        load_count += 1
        return sentinel

    monkeypatch.setattr(promotion_policy_v3, "_neo4j_driver_from_env", _fake_loader)
    monkeypatch.setattr(promotion_policy_v3, "_NEO4J_DRIVER_SINGLETON", None)

    first = promotion_policy_v3._shared_neo4j_driver()
    second = promotion_policy_v3._shared_neo4j_driver()

    assert first is sentinel
    assert second is sentinel
    assert load_count == 1

    promotion_policy_v3._close_shared_neo4j_driver()
    assert sentinel.closed is True
