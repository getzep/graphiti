"""Unit tests for ``EpisodicNode.get_recent_by_group_ids``.

``get_by_group_ids`` orders by uuid because it backs keyset pagination. Browsing
callers that want the most recent episodes need a time ordering, and because
``LIMIT`` is applied inside the query they cannot get one by re-sorting the page
they were handed. These tests lock in the emitted ``ORDER BY``, the uuid
tiebreak, and the fact that the pagination helper is left alone.

They use a ``spec=GraphDriver`` mock, so they need no database and run in CI.
"""

from datetime import datetime, timedelta, timezone
from unittest.mock import AsyncMock, MagicMock

import pytest

from graphiti_core.driver.driver import GraphDriver, GraphProvider
from graphiti_core.nodes import EpisodicNode

pytestmark = pytest.mark.asyncio

GROUP_IDS = ['group-a']


def _make_driver(provider=GraphProvider.FALKORDB, records=None):
    driver = MagicMock(spec=GraphDriver)
    driver.provider = provider
    driver.graph_operations_interface = None
    driver.execute_query = AsyncMock(return_value=(records or [], None, None))
    return driver


def _episode_record(uuid: str, created_at: datetime, valid_at: datetime) -> dict:
    return {
        'uuid': uuid,
        'name': f'Episode {uuid}',
        'group_id': GROUP_IDS[0],
        'created_at': created_at,
        'valid_at': valid_at,
        'source': 'text',
        'source_description': 'test',
        'content': 'some content',
        'entity_edges': [],
    }


def _emitted_query(driver) -> str:
    return driver.execute_query.await_args.args[0]


def _emitted_kwargs(driver) -> dict:
    return driver.execute_query.await_args.kwargs


async def test_defaults_to_valid_at_with_uuid_tiebreak():
    driver = _make_driver()

    await EpisodicNode.get_recent_by_group_ids(driver, GROUP_IDS)

    query = _emitted_query(driver)
    assert 'ORDER BY valid_at DESC, uuid DESC' in query


async def test_created_at_ordering():
    driver = _make_driver()

    await EpisodicNode.get_recent_by_group_ids(driver, GROUP_IDS, order_by='created_at')

    assert 'ORDER BY created_at DESC, uuid DESC' in _emitted_query(driver)


async def test_uuid_ordering_remains_available():
    driver = _make_driver()

    await EpisodicNode.get_recent_by_group_ids(driver, GROUP_IDS, order_by='uuid')

    query = _emitted_query(driver)
    assert 'ORDER BY uuid DESC' in query
    assert 'valid_at DESC' not in query
    assert 'created_at DESC' not in query


async def test_invalid_order_by_raises_before_querying():
    driver = _make_driver()

    with pytest.raises(ValueError, match='Invalid order_by'):
        await EpisodicNode.get_recent_by_group_ids(driver, GROUP_IDS, order_by='name')  # type: ignore[arg-type]

    driver.execute_query.assert_not_called()


async def test_limit_is_applied_in_query_not_client_side():
    """The LIMIT has to sit inside the ordered query — that is why a caller cannot
    fix the ordering by re-sorting the page it received."""
    driver = _make_driver()

    await EpisodicNode.get_recent_by_group_ids(driver, GROUP_IDS, limit=5)

    query = _emitted_query(driver)
    assert 'LIMIT $limit' in query
    assert query.index('ORDER BY') < query.index('LIMIT $limit')
    assert _emitted_kwargs(driver)['limit'] == 5


async def test_no_limit_clause_when_limit_omitted():
    driver = _make_driver()

    await EpisodicNode.get_recent_by_group_ids(driver, GROUP_IDS)

    assert 'LIMIT' not in _emitted_query(driver)


async def test_takes_no_cursor():
    """This is a browsing read, not a paged one: no cursor predicate is emitted."""
    driver = _make_driver()

    await EpisodicNode.get_recent_by_group_ids(driver, GROUP_IDS)

    query = _emitted_query(driver)
    assert 'e.uuid <' not in query
    assert 'uuid' not in _emitted_kwargs(driver)


async def test_routes_reads_to_a_replica():
    driver = _make_driver()

    await EpisodicNode.get_recent_by_group_ids(driver, GROUP_IDS)

    assert _emitted_kwargs(driver)['routing_'] == 'r'
    assert _emitted_kwargs(driver)['group_ids'] == GROUP_IDS


async def test_neptune_keeps_its_return_clause():
    driver = _make_driver(provider=GraphProvider.NEPTUNE)

    await EpisodicNode.get_recent_by_group_ids(driver, GROUP_IDS)

    query = _emitted_query(driver)
    assert 'split(e.entity_edges' in query
    assert 'ORDER BY valid_at DESC, uuid DESC' in query


async def test_records_are_returned_in_driver_order():
    """No client-side re-sort: the database's ordering is the answer."""
    now = datetime.now(timezone.utc)
    records = [
        _episode_record('zzz', created_at=now - timedelta(days=9), valid_at=now),
        _episode_record('aaa', created_at=now, valid_at=now - timedelta(days=9)),
    ]
    driver = _make_driver(records=records)

    episodes = await EpisodicNode.get_recent_by_group_ids(driver, GROUP_IDS)

    assert [episode.uuid for episode in episodes] == ['zzz', 'aaa']


async def test_pagination_helper_still_orders_by_uuid():
    """Regression guard: the keyset helper and its cursor are deliberately untouched."""
    driver = _make_driver()

    await EpisodicNode.get_by_group_ids(driver, GROUP_IDS, limit=5, uuid_cursor='some-uuid')

    query = _emitted_query(driver)
    assert 'ORDER BY uuid DESC' in query
    assert 'AND e.uuid < $uuid' in query
    assert 'valid_at DESC' not in query


async def test_delegates_to_the_driver_operations_interface_when_present():
    driver = _make_driver()
    driver.graph_operations_interface = MagicMock()
    driver.graph_operations_interface.episodic_node_get_recent_by_group_ids = AsyncMock(
        return_value=[]
    )

    result = await EpisodicNode.get_recent_by_group_ids(
        driver, GROUP_IDS, limit=3, order_by='created_at'
    )

    assert result == []
    driver.graph_operations_interface.episodic_node_get_recent_by_group_ids.assert_awaited_once_with(
        EpisodicNode, driver, GROUP_IDS, 3, 'created_at'
    )
    driver.execute_query.assert_not_called()


async def test_falls_back_to_generic_query_when_interface_does_not_implement_it():
    driver = _make_driver()
    driver.graph_operations_interface = MagicMock()
    driver.graph_operations_interface.episodic_node_get_recent_by_group_ids = AsyncMock(
        side_effect=NotImplementedError
    )

    await EpisodicNode.get_recent_by_group_ids(driver, GROUP_IDS)

    assert 'ORDER BY valid_at DESC, uuid DESC' in _emitted_query(driver)
