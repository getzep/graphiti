from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from mcp_server.src.services.neo4j_service import (
    OM_FULLTEXT_CREATE_ONLINE_MAX_ATTEMPTS,
    Neo4jService,
    _fact_relation_expansion_limit,
)


class _Record(dict):
    def data(self):
        return dict(self)


def _index_payload(state: str) -> dict[str, object]:
    return {
        "name": "omnode_content_fulltext",
        "type": "FULLTEXT",
        "state": state,
        "entityType": "NODE",
        "labelsOrTypes": ["OMNode"],
        "properties": ["content", "group_id"],
    }


def test_verify_fulltext_index_missing_creates_with_write_routing() -> None:
    service = Neo4jService()
    driver = MagicMock()

    lookup_states = [None, _index_payload("ONLINE")]
    calls: list[dict[str, object]] = []

    async def _execute_query(query, *args, routing_="r", **kwargs):
        calls.append({"query": query, "routing": routing_, "kwargs": kwargs})

        if "SHOW INDEXES" in query:
            payload = lookup_states.pop(0)
            if payload is None:
                return [], [], None
            return [_Record(payload)], [], None

        if "CREATE FULLTEXT INDEX" in query:
            return [], [], None

        raise AssertionError(f"Unexpected query: {query}")

    driver.execute_query = AsyncMock(side_effect=_execute_query)

    asyncio.run(service.verify_om_fulltext_index_shape(driver))

    create_calls = [
        c for c in calls if "CREATE FULLTEXT INDEX" in str(c["query"])
    ]
    assert len(create_calls) == 1
    assert create_calls[0]["routing"] == "w"

    create_idx = calls.index(create_calls[0])
    post_create_show_calls = [
        c for c in calls[create_idx + 1 :] if "SHOW INDEXES" in str(c["query"])
    ]
    assert post_create_show_calls
    assert all(c["routing"] == "w" for c in post_create_show_calls)


def test_verify_fulltext_index_waits_through_populating_after_create() -> None:
    service = Neo4jService()
    driver = MagicMock()

    lookup_states = [
        None,
        _index_payload("POPULATING"),
        _index_payload("POPULATING"),
        _index_payload("ONLINE"),
    ]

    async def _execute_query(query, *args, routing_="r", **kwargs):
        if "SHOW INDEXES" in query:
            payload = lookup_states.pop(0)
            if payload is None:
                return [], [], None
            return [_Record(payload)], [], None

        if "CREATE FULLTEXT INDEX" in query:
            return [], [], None

        raise AssertionError(f"Unexpected query: {query}")

    driver.execute_query = AsyncMock(side_effect=_execute_query)

    with patch("mcp_server.src.services.neo4j_service.asyncio.sleep", new=AsyncMock()) as sleep_mock:
        asyncio.run(service.verify_om_fulltext_index_shape(driver))

    assert sleep_mock.await_count == 2


def test_verify_fulltext_index_fails_loud_when_never_online_after_create() -> None:
    service = Neo4jService()
    driver = MagicMock()

    lookup_states = [None] + [_index_payload("POPULATING")] * OM_FULLTEXT_CREATE_ONLINE_MAX_ATTEMPTS

    async def _execute_query(query, *args, routing_="r", **kwargs):
        if "SHOW INDEXES" in query:
            payload = lookup_states.pop(0)
            if payload is None:
                return [], [], None
            return [_Record(payload)], [], None

        if "CREATE FULLTEXT INDEX" in query:
            return [], [], None

        raise AssertionError(f"Unexpected query: {query}")

    driver.execute_query = AsyncMock(side_effect=_execute_query)

    with (
        patch("mcp_server.src.services.neo4j_service.asyncio.sleep", new=AsyncMock()),
        pytest.raises(RuntimeError) as exc_info,
    ):
        asyncio.run(service.verify_om_fulltext_index_shape(driver))

    message = str(exc_info.value)
    assert "timed out waiting for index to reach ONLINE after create attempt" in message
    assert "state=POPULATING" in message


def test_search_om_nodes_lexical_fulltext_enforces_group_id_filter() -> None:
    service = Neo4jService()
    driver = MagicMock()
    captured: dict[str, object] = {}

    async def _execute_query(query, *args, routing_="r", **kwargs):
        captured["query"] = query
        captured["routing"] = routing_
        captured["kwargs"] = kwargs
        return [], [], None

    driver.execute_query = AsyncMock(side_effect=_execute_query)

    result = asyncio.run(
        service.search_om_nodes(
            driver,
            group_id="s1_observational_memory",
            query="index hardening fix",
            limit=7,
        )
    )

    assert result == []
    assert captured["routing"] == "r"
    assert "AND node.group_id = $group_id" in str(captured["query"])


def test_search_om_facts_bounds_per_candidate_relationship_expansion() -> None:
    service = Neo4jService()
    driver = MagicMock()
    captured: dict[str, object] = {}

    async def _execute_query(query, *args, routing_="r", **kwargs):
        captured["query"] = query
        captured["routing"] = routing_
        captured["kwargs"] = kwargs
        return [], [], None

    driver.execute_query = AsyncMock(side_effect=_execute_query)

    result = asyncio.run(
        service.search_om_facts(
            driver,
            group_id="s1_observational_memory",
            query="index hardening fix",
            limit=7,
            center_node_uuid=None,
        )
    )

    assert result == []
    assert captured["routing"] == "r"
    assert "AND matched_node.group_id = $group_id" in str(captured["query"])
    assert "LIMIT $per_candidate_relationship_limit" in str(captured["query"])
    assert captured["kwargs"]["per_candidate_relationship_limit"] == _fact_relation_expansion_limit(7)
