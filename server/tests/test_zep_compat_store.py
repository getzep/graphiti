"""Store tests for the invariants MiroFish's reconciliation logic depends on.

No graph database, no LLM, no network — SQLite only.

Run:  uv run --extra dev pytest tests/test_zep_compat_store.py
"""

from __future__ import annotations

import pytest

from graph_service.zep_compat.store import Store


@pytest.fixture
def store(tmp_path):
    s = Store(tmp_path / 'compat.sqlite3')
    yield s
    s.close()


def _items(count, graph_id='g-1', start=0):
    return [
        {'graph_id': graph_id, 'payload': f'chunk-{i}', 'data_type': 'text'}
        for i in range(start, start + count)
    ]


# ---------------------------------------------------------------------------
# graphs
# ---------------------------------------------------------------------------


def test_unknown_graph_is_none_so_the_route_can_404(store):
    assert store.get_graph('nope') is None


def test_create_graph_is_idempotent(store):
    first = store.create_graph('g-1', 'n', 'd', 'UTC')
    second = store.create_graph('g-1', 'other', 'other', 'UTC')
    # Same identity on repeat create; MiroFish may retry after a lost response.
    assert first['uuid'] == second['uuid']
    assert second['name'] == 'n'


def test_delete_graph_removes_ontology_too(store):
    store.create_graph('g-1', None, None, None)
    store.set_ontology('g-1', [{'name': 'T', 'description': 'd'}], [])
    store.delete_graph('g-1')
    assert store.get_graph('g-1') is None
    assert store.get_ontology('g-1') == ([], [])


def test_ontology_upsert_replaces(store):
    store.set_ontology('g-1', [{'name': 'A', 'description': 'd'}], [])
    store.set_ontology('g-1', [{'name': 'B', 'description': 'd'}], [])
    entities, edges = store.get_ontology('g-1')
    assert [e['name'] for e in entities] == ['B']
    assert edges == []


# ---------------------------------------------------------------------------
# batch item indexing — the reconciliation contract
# ---------------------------------------------------------------------------


def test_sequence_index_is_global_across_multiple_add_calls(store):
    """add_text_batches asserts recovered indexes == set(range(expected)).

    So indexes must be global across the whole batch, not per-request.
    """
    batch_id = store.create_batch({'mirofish_operation_id': 'op-1'}, None)
    first = store.add_items(batch_id, _items(3))
    second = store.add_items(batch_id, _items(2, start=3))

    assert [i['sequence_index'] for i in first] == [0, 1, 2]
    assert [i['sequence_index'] for i in second] == [3, 4]

    all_items, _ = store.list_items(batch_id, limit=100, cursor=None)
    assert {i['sequence_index'] for i in all_items} == set(range(5))


def test_episode_uuid_assigned_at_add_time_and_is_unique(store):
    """MiroFish reads episode_uuid off the *add* response, before processing."""
    batch_id = store.create_batch(None, None)
    created = store.add_items(batch_id, _items(4))
    uuids = [i['episode_uuid'] for i in created]
    assert all(uuids)
    assert len(set(uuids)) == 4


def test_episode_uuid_is_stable_between_add_and_list(store):
    batch_id = store.create_batch(None, None)
    created = store.add_items(batch_id, _items(2))
    listed, _ = store.list_items(batch_id, limit=100, cursor=None)
    assert [i['episode_uuid'] for i in created] == [i['episode_uuid'] for i in listed]


# ---------------------------------------------------------------------------
# pagination — MiroFish raises if a cursor fails to advance
# ---------------------------------------------------------------------------


def test_item_cursor_advances_and_terminates(store):
    batch_id = store.create_batch(None, None)
    store.add_items(batch_id, _items(250))

    seen, cursor, seen_cursors = [], None, set()
    while True:
        page, next_cursor = store.list_items(batch_id, limit=100, cursor=cursor)
        seen.extend(page)
        if next_cursor is None:
            break
        assert next_cursor != cursor, 'cursor must advance or MiroFish raises'
        assert next_cursor not in seen_cursors
        seen_cursors.add(next_cursor)
        cursor = next_cursor

    assert len(seen) == 250
    assert {i['sequence_index'] for i in seen} == set(range(250))


def test_item_last_page_returns_no_cursor(store):
    batch_id = store.create_batch(None, None)
    store.add_items(batch_id, _items(5))
    page, next_cursor = store.list_items(batch_id, limit=100, cursor=None)
    assert len(page) == 5
    assert next_cursor is None


def test_exact_page_boundary_does_not_emit_a_dangling_cursor(store):
    """A cursor on an exactly-full final page would make MiroFish fetch an
    empty page; harmless but it must still terminate."""
    batch_id = store.create_batch(None, None)
    store.add_items(batch_id, _items(100))
    page, next_cursor = store.list_items(batch_id, limit=100, cursor=None)
    assert len(page) == 100
    assert next_cursor is None


def test_batch_list_cursor_advances_and_finds_by_metadata(store):
    """_find_batch_by_operation_id pages batch.list matching metadata."""
    target = None
    for i in range(150):
        meta = {'mirofish_operation_id': f'op-{i}', 'graph_id': 'g-1'}
        batch_id = store.create_batch(meta, None)
        if i == 120:
            target = batch_id

    matches, cursor, seen_cursors = [], None, set()
    while True:
        page, next_cursor = store.list_batches(limit=100, cursor=cursor, status=None)
        matches.extend(
            b
            for b in page
            if (b['metadata'] or {}).get('mirofish_operation_id') == 'op-120'
            and (b['metadata'] or {}).get('graph_id') == 'g-1'
        )
        if next_cursor is None:
            break
        assert next_cursor != cursor
        assert next_cursor not in seen_cursors
        seen_cursors.add(next_cursor)
        cursor = next_cursor

    # Exactly one match, or MiroFish raises "refusing ambiguity".
    assert len(matches) == 1
    assert matches[0]['batch_id'] == target


def test_batch_list_status_filter(store):
    a = store.create_batch({'k': 'a'}, None)
    store.create_batch({'k': 'b'}, None)
    store.set_batch_status(a, 'succeeded', completed=True)
    page, _ = store.list_batches(limit=100, cursor=None, status='succeeded')
    assert [b['batch_id'] for b in page] == [a]


# ---------------------------------------------------------------------------
# status lifecycle
# ---------------------------------------------------------------------------


def test_new_batch_is_draft(store):
    """_wait_for_batch's reconciliation treats None/draft as 'not processed'."""
    batch_id = store.create_batch(None, None)
    assert store.get_batch(batch_id)['status'] == 'draft'


def test_counts_track_item_transitions(store):
    batch_id = store.create_batch(None, None)
    created = store.add_items(batch_id, _items(4))
    assert store.item_counts(batch_id) == {'pending': 4, 'total': 4}

    store.mark_items_queued(batch_id)
    assert store.item_counts(batch_id)['queued'] == 4

    store.set_item_status(created[0]['item_id'], 'succeeded')
    store.set_item_status(created[1]['item_id'], 'failed', {'message': 'boom'})
    counts = store.item_counts(batch_id)
    assert counts['succeeded'] == 1
    assert counts['failed'] == 1
    assert counts['total'] == 4


def test_failed_item_error_survives_listing(store):
    """_wait_for_batch surfaces the first failed item's error."""
    batch_id = store.create_batch(None, None)
    created = store.add_items(batch_id, _items(1))
    store.set_item_status(created[0]['item_id'], 'failed', {'message': 'llm timeout'})
    items, _ = store.list_items(batch_id, limit=10, cursor=None)
    assert items[0]['error'] == {'message': 'llm timeout'}


def test_pending_items_excludes_settled_work(store):
    batch_id = store.create_batch(None, None)
    created = store.add_items(batch_id, _items(3))
    store.set_item_status(created[0]['item_id'], 'succeeded')
    store.set_item_status(created[1]['item_id'], 'skipped')
    pending = store.pending_items(batch_id)
    assert [i['item_id'] for i in pending] == [created[2]['item_id']]


def test_failed_items_are_retried_on_resume(store):
    """A crash mid-batch must not silently drop failed items."""
    batch_id = store.create_batch(None, None)
    created = store.add_items(batch_id, _items(2))
    store.set_item_status(created[0]['item_id'], 'failed', {'message': 'x'})
    assert {i['item_id'] for i in store.pending_items(batch_id)} == {
        created[0]['item_id'],
        created[1]['item_id'],
    }


def test_claim_draft_batches_finds_only_in_flight_work(store):
    draft = store.create_batch(None, None)
    queued = store.create_batch(None, None)
    processing = store.create_batch(None, None)
    done = store.create_batch(None, None)
    store.set_batch_status(queued, 'queued', processed=True)
    store.set_batch_status(processing, 'processing')
    store.set_batch_status(done, 'succeeded', completed=True)

    claimed = set(store.claim_draft_batches())
    assert claimed == {queued, processing}
    # A draft was never handed to /process, so resuming it would be wrong.
    assert draft not in claimed


def test_state_survives_reopen(tmp_path):
    """Restart durability: MiroFish reconciles a batch after a lost response."""
    path = tmp_path / 'compat.sqlite3'
    first = Store(path)
    batch_id = first.create_batch({'mirofish_operation_id': 'op-1'}, None)
    created = first.add_items(batch_id, _items(2))
    first.set_batch_status(batch_id, 'processing')
    first.close()

    second = Store(path)
    try:
        record = second.get_batch(batch_id)
        assert record is not None
        assert record['status'] == 'processing'
        assert record['metadata']['mirofish_operation_id'] == 'op-1'
        items, _ = second.list_items(batch_id, limit=10, cursor=None)
        assert [i['episode_uuid'] for i in items] == [
            i['episode_uuid'] for i in created
        ]
    finally:
        second.close()
