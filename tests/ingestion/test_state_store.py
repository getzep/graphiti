"""Conformance tests for `IngestionStateStore`.

`StoreConformance` takes a store factory. A host application can run the same class
for its store: make a subclass and override the `make_store` fixture.
"""

import multiprocessing
from collections.abc import Callable

import pytest

from graphiti_core.errors import StaleCommitError, UnitExpiredError
from graphiti_core.ingestion.sqlite_state_store import SQLiteIngestionStateStore
from graphiti_core.ingestion.state_store import (
    CommitRecover,
    CommitTurn,
    CommitWait,
    IngestionStateStore,
    InMemoryIngestionStateStore,
)

G = 'group-1'
TTL = 60.0


class FakeClock:
    def __init__(self) -> None:
        self.now = 1_000.0

    def __call__(self) -> float:
        return self.now

    def advance(self, seconds: float) -> None:
        self.now += seconds


async def admit(store: IngestionStateStore, unit_id: str, group_id: str = G) -> int:
    admission = await store.begin_admit(group_id, unit_id, TTL)
    assert admission is not None
    await store.end_admit(group_id, admission.seq)
    return admission.seq


async def commit(
    store: IngestionStateStore,
    seq: int,
    payload: bytes = b'final',
    node_uuids: list[str] | None = None,
    retain_seconds: float = 0.0,
) -> int:
    turn = await store.begin_commit(G, seq, TTL)
    assert isinstance(turn, CommitTurn)
    await store.save_final(G, seq, turn.fence, payload)
    await store.finish_commit(G, seq, turn.fence, node_uuids or [], retain_seconds)
    return turn.fence


class StoreConformance:
    @pytest.fixture
    def clock(self) -> FakeClock:
        return FakeClock()

    @pytest.fixture
    def make_store(self, clock: FakeClock, tmp_path) -> Callable[[], IngestionStateStore]:
        raise NotImplementedError

    @pytest.fixture
    async def store(self, make_store):
        store = make_store()
        yield store
        await store.close()

    async def test_sequence_numbers_start_at_one_and_have_no_gap(self, store):
        assert [await admit(store, f'u{i}') for i in range(4)] == [1, 2, 3, 4]

    async def test_same_unit_id_gets_same_seq(self, store):
        first = await store.begin_admit(G, 'u1', TTL)
        second = await store.begin_admit(G, 'u1', TTL)
        assert first == second
        await store.end_admit(G, first.seq)
        assert (await store.begin_admit(G, 'u1', TTL)).seq == first.seq

    async def test_groups_have_separate_sequences(self, store):
        assert await admit(store, 'u1', 'a') == 1
        assert await admit(store, 'u1', 'b') == 1

    async def test_no_admit_lease_before_end_admit(self, store):
        first = await store.begin_admit(G, 'u1', TTL)
        assert await store.begin_admit(G, 'u2', TTL) is None
        await store.end_admit(G, first.seq)
        assert (await store.begin_admit(G, 'u2', TTL)).seq == 2

    async def test_expired_admission_releases_lease_and_fails(self, store, clock):
        first = await store.begin_admit(G, 'u1', TTL)
        clock.advance(TTL + 1)
        second = await store.begin_admit(G, 'u2', TTL)
        assert second.seq == 2
        with pytest.raises(UnitExpiredError):
            await store.end_admit(G, first.seq)
        await store.end_admit(G, second.seq)
        # The failed unit does not hold the commit order.
        assert isinstance(await store.begin_commit(G, second.seq, TTL), CommitTurn)

    async def test_start_analysis_returns_watermark_and_time(self, store, clock):
        seq1 = await admit(store, 'u1')
        seq2 = await admit(store, 'u2')
        assert (await store.start_analysis(G, seq2)).snapshot_seq == 0
        await commit(store, seq1)
        seq3 = await admit(store, 'u3')
        start = await store.start_analysis(G, seq3)
        assert start.snapshot_seq == 1
        assert start.started_at == clock.now

    async def test_commit_wait_then_turn(self, store):
        seq1 = await admit(store, 'u1')
        seq2 = await admit(store, 'u2')
        assert isinstance(await store.begin_commit(G, seq2, TTL), CommitWait)
        await commit(store, seq1)
        assert isinstance(await store.begin_commit(G, seq2, TTL), CommitTurn)

    async def test_commit_turn_after_earlier_unit_failed(self, store):
        seq1 = await admit(store, 'u1')
        seq2 = await admit(store, 'u2')
        await store.fail(G, seq1, 'error')
        turn = await store.begin_commit(G, seq2, TTL)
        assert isinstance(turn, CommitTurn)
        assert turn.final_payload is None

    async def test_expired_earlier_unit_with_no_final_becomes_failed(self, store, clock):
        seq1 = await admit(store, 'u1')
        await store.save_change_set(G, seq1, b'analysis')
        clock.advance(TTL - 1)
        seq2 = await admit(store, 'u2')
        clock.advance(2)
        assert isinstance(await store.begin_commit(G, seq2, TTL), CommitTurn)
        with pytest.raises(UnitExpiredError):
            await store.heartbeat(G, seq1, TTL)
        assert await store.load_delta(G, 0, 10, 0.0) == []

    async def test_expired_applying_unit_gives_commit_recover(self, store, clock):
        seq1 = await admit(store, 'u1')
        turn1 = await store.begin_commit(G, seq1, TTL)
        await store.save_final(G, seq1, turn1.fence, b'final-1')
        clock.advance(TTL - 1)
        seq2 = await admit(store, 'u2')
        assert isinstance(await store.begin_commit(G, seq2, TTL), CommitWait)
        clock.advance(2)
        recover = await store.begin_commit(G, seq2, TTL)
        assert isinstance(recover, CommitRecover)
        assert (recover.seq, recover.final_payload) == (seq1, b'final-1')
        assert recover.fence == turn1.fence + 1

        # The old attempt is stale.
        with pytest.raises(StaleCommitError):
            await store.save_final(G, seq1, turn1.fence, b'other')
        with pytest.raises(StaleCommitError):
            await store.finish_commit(G, seq1, turn1.fence, [], 0.0)
        with pytest.raises(StaleCommitError):
            await store.fail(G, seq1, 'apply_failed', turn1.fence)

        await store.finish_commit(G, seq1, recover.fence, [], 0.0)
        assert isinstance(await store.begin_commit(G, seq2, TTL), CommitTurn)

    async def test_own_retry_gets_stored_final_payload(self, store, clock):
        seq1 = await admit(store, 'u1')
        turn = await store.begin_commit(G, seq1, TTL)
        await store.save_final(G, seq1, turn.fence, b'final-1')
        clock.advance(TTL + 1)
        retry = await store.begin_commit(G, seq1, TTL)
        assert isinstance(retry, CommitTurn)
        assert retry.final_payload == b'final-1'
        assert retry.fence == turn.fence + 1

    async def test_fail_is_not_permitted_for_applying(self, store):
        seq1 = await admit(store, 'u1')
        turn = await store.begin_commit(G, seq1, TTL)
        await store.save_final(G, seq1, turn.fence, b'final-1')
        with pytest.raises(ValueError):
            await store.fail(G, seq1, 'error')
        with pytest.raises(StaleCommitError):
            await store.fail(G, seq1, 'apply_failed')

    async def test_apply_failed_keeps_payload_and_releases_position(self, store, clock):
        seq1 = await admit(store, 'u1')
        turn = await store.begin_commit(G, seq1, TTL)
        await store.save_final(G, seq1, turn.fence, b'final-1')
        seq2 = await admit(store, 'u2')
        await store.fail(G, seq1, 'apply_failed', turn.fence)

        assert isinstance(await store.begin_commit(G, seq2, TTL), CommitTurn)
        delta = await store.load_delta(G, 0, seq2, clock.now)
        assert [(unit.seq, unit.status, unit.payload) for unit in delta] == [
            (seq1, 'apply_failed', b'final-1')
        ]
        with pytest.raises(UnitExpiredError):
            await store.begin_commit(G, seq1, TTL)

    async def test_load_delta_range_and_since(self, store, clock):
        seqs = [await admit(store, f'u{i}') for i in range(4)]
        await commit(store, seqs[0], b'p1', retain_seconds=1_000)
        clock.advance(30)
        await commit(store, seqs[1], b'p2', retain_seconds=1_000)
        await commit(store, seqs[2], b'p3', retain_seconds=1_000)

        delta = await store.load_delta(G, 2, 4, clock.now + 1)
        assert [(unit.seq, unit.status, unit.payload) for unit in delta] == [
            (3, 'committed', b'p3')
        ]
        # Units at or below after_seq are in the result when they ended at or after `since`.
        delta = await store.load_delta(G, 2, 4, clock.now - 10)
        assert [unit.seq for unit in delta] == [2, 3]
        delta = await store.load_delta(G, 2, 4, 0.0)
        assert [unit.seq for unit in delta] == [1, 2, 3]
        # before_seq is exclusive.
        assert [unit.seq for unit in await store.load_delta(G, 0, 3, clock.now + 1)] == [1, 2]

    async def test_analysis_payload_is_not_in_delta(self, store):
        seq1 = await admit(store, 'u1')
        await store.save_change_set(G, seq1, b'analysis')
        seq2 = await admit(store, 'u2')
        assert await store.load_delta(G, 0, seq2, 0.0) == []

    async def test_ready_enrich_in_seq_order_for_each_node(self, store):
        seq1 = await admit(store, 'u1')
        seq2 = await admit(store, 'u2')
        await commit(store, seq1, node_uuids=['a', 'b'])
        await commit(store, seq2, node_uuids=['b', 'c'])

        assert await store.ready_enrich(G, seq1) == ['a', 'b']
        assert await store.ready_enrich(G, seq2) == ['c']
        await store.finish_enrich(G, seq1, ['b'])
        assert await store.ready_enrich(G, seq2) == ['b', 'c']
        await store.finish_enrich(G, seq2, ['b', 'c'])
        assert await store.ready_enrich(G, seq2) == []
        assert await store.ready_enrich(G, seq1) == ['a']

    async def test_entries_of_expired_unit_do_not_stop_later_unit(self, store, clock):
        seq1 = await admit(store, 'u1')
        seq2 = await admit(store, 'u2')
        await commit(store, seq1, node_uuids=['a'])
        clock.advance(TTL - 1)
        await store.heartbeat(G, seq2, TTL)
        await commit(store, seq2, node_uuids=['a'])
        assert await store.ready_enrich(G, seq2) == []
        clock.advance(2)
        assert await store.ready_enrich(G, seq2) == ['a']
        with pytest.raises(UnitExpiredError):
            await store.ready_enrich(G, seq1)

    async def test_fail_deletes_enrichment_entries_only_before_commit(self, store):
        seq1 = await admit(store, 'u1')
        await commit(store, seq1, node_uuids=['a'])
        # `fail` for a committed unit changes nothing.
        await store.fail(G, seq1, 'error')
        assert await store.ready_enrich(G, seq1) == ['a']

    async def test_expired_unit_raises(self, store, clock):
        seq1 = await admit(store, 'u1')
        clock.advance(TTL + 1)
        for call in (
            lambda: store.start_analysis(G, seq1),
            lambda: store.save_change_set(G, seq1, b'x'),
            lambda: store.heartbeat(G, seq1, TTL),
            lambda: store.begin_commit(G, seq1, TTL),
            lambda: store.ready_enrich(G, seq1),
        ):
            with pytest.raises(UnitExpiredError):
                await call()

    async def test_failed_unit_raises(self, store):
        seq1 = await admit(store, 'u1')
        await store.fail(G, seq1, 'error')
        with pytest.raises(UnitExpiredError):
            await store.start_analysis(G, seq1)
        with pytest.raises(UnitExpiredError):
            await store.save_final(G, seq1, 1, b'x')

    async def test_heartbeat_extends_deadline(self, store, clock):
        seq1 = await admit(store, 'u1')
        clock.advance(TTL - 1)
        await store.heartbeat(G, seq1, TTL)
        clock.advance(TTL - 1)
        await store.save_change_set(G, seq1, b'x')

    async def test_old_units_are_deleted_only_with_three_conditions(self, store, clock):
        retain = 300.0
        seq1 = await admit(store, 'u1')
        seq2 = await admit(store, 'u2')
        seq3 = await admit(store, 'u3')
        for seq in (seq1, seq2, seq3):
            await store.heartbeat(G, seq, retain * 10)
        # Unit 3 is in progress with snapshot 0.
        await store.start_analysis(G, seq3)
        await commit(store, seq1, b'p1', node_uuids=['a'], retain_seconds=retain)

        async def delta_seqs() -> list[int]:
            return [unit.seq for unit in await store.load_delta(G, 0, 100, 0.0)]

        clock.advance(retain + 1)
        await commit(store, seq2, b'p2', retain_seconds=retain)
        # Unit 1 is old, but it has an enrichment entry and a snapshot is below it.
        assert await delta_seqs() == [1, 2]

        await store.finish_enrich(G, seq1, ['a'])
        await commit(store, seq3, b'p3', retain_seconds=retain)
        # No unit in progress and no entry: unit 1 is deleted. Units 2 and 3 are not old.
        assert await delta_seqs() == [2, 3]

        clock.advance(retain + 1)
        seq4 = await admit(store, 'u4')
        await commit(store, seq4, b'p4', retain_seconds=retain)
        assert await delta_seqs() == [4]
        assert seq4 == 4


class TestInMemoryStore(StoreConformance):
    @pytest.fixture
    def make_store(self, clock, tmp_path):
        return lambda: InMemoryIngestionStateStore(clock=clock)


class TestSQLiteStore(StoreConformance):
    @pytest.fixture
    def make_store(self, clock, tmp_path):
        path = str(tmp_path / 'ingest.db')
        return lambda: SQLiteIngestionStateStore(path, clock=clock)

    async def test_two_store_objects_on_one_file(self, make_store):
        first, second = make_store(), make_store()
        try:
            seq1 = await admit(first, 'u1')
            seq2 = await admit(second, 'u2')
            assert (seq1, seq2) == (1, 2)
            assert isinstance(await second.begin_commit(G, seq2, TTL), CommitWait)
            await commit(first, seq1, b'p1', retain_seconds=1_000)
            assert isinstance(await second.begin_commit(G, seq2, TTL), CommitTurn)
            assert [unit.payload for unit in await second.load_delta(G, 0, seq2, 0.0)] == [b'p1']
        finally:
            await first.close()
            await second.close()

    async def test_second_process_on_one_file(self, tmp_path):
        path = str(tmp_path / 'shared.db')
        store = SQLiteIngestionStateStore(path)
        try:
            seq1 = await admit(store, 'parent-unit')
            await commit(store, seq1, b'parent', retain_seconds=1_000)
            context = multiprocessing.get_context('spawn')
            with context.Pool(1) as pool:
                child_seq, child_delta = pool.apply(_child_process_work, (path,))
            assert child_seq == 2
            assert child_delta == [b'parent']
            assert [unit.payload for unit in await store.load_delta(G, 0, 10, 0.0)] == [
                b'parent',
                b'child',
            ]
        finally:
            await store.close()


def _child_process_work(path: str) -> tuple[int, list[bytes]]:
    import asyncio

    async def work() -> tuple[int, list[bytes]]:
        store = SQLiteIngestionStateStore(path)
        try:
            seq = await admit(store, 'child-unit')
            delta = [unit.payload for unit in await store.load_delta(G, 0, seq, 0.0)]
            await commit(store, seq, b'child', retain_seconds=1_000)
            return seq, delta
        finally:
            await store.close()

    return asyncio.run(work())
