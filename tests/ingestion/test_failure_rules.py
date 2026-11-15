"""
Copyright 2024, Zep Software, Inc.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

# Tests for the failure rules and for the Enrich phase of concurrent ingestion.

import asyncio
from unittest.mock import AsyncMock

import pytest

from graphiti_core.errors import StaleCommitError, UnitExpiredError
from graphiti_core.ingestion import phases
from graphiti_core.ingestion.change_set import ChangeSet, PatchNode
from graphiti_core.ingestion.phases import UnitLease, admit_episode, analyze_episode
from graphiti_core.ingestion.state_store import (
    STATUS_APPLY_FAILED,
    STATUS_COMMITTED,
    STATUS_FAILED,
    CommitWait,
)
from tests.ingestion.ingestion_fixtures import (
    GROUP,
    T0,
    FakeClock,
    Harness,
    Y,
    make_node,
)


def status(h: Harness, seq: int) -> str:
    return h.store._groups[GROUP].units[seq].status


# --- failures before the final change set is stored -------------------------------


async def test_analysis_failure_releases_the_position(monkeypatch):
    h = Harness(monkeypatch)
    monkeypatch.setattr(phases, 'extract_nodes', AsyncMock(side_effect=RuntimeError('model')))
    unit_1 = await h.admit('u1')

    with pytest.raises(RuntimeError):
        await analyze_episode(h.graphiti, h.store, h.config, unit_1)

    assert status(h, unit_1.seq) == STATUS_FAILED
    assert h.graph.nodes == {}
    # A later unit does not wait for the failed unit.
    _, result_2 = await h.run('u2', create_nodes=[make_node('a2', 'Acme')])
    assert result_2.report['turn_wait_ms'] < 1000
    assert set(h.graph.nodes) == {'a2'}


async def test_admission_lease_expiry_fails_the_holder(monkeypatch):
    clock = FakeClock()
    h = Harness(monkeypatch, clock=clock)
    h.graphiti.retrieve_episodes = AsyncMock(return_value=[])
    # Unit 1 gets the admit lease and stops.
    stopped = await h.store.begin_admit(GROUP, 'u1', h.config.lease_seconds)
    assert await h.store.begin_admit(GROUP, 'u2', h.config.lease_seconds) is None
    clock.advance(h.config.lease_seconds + 1)

    unit_2 = await admit_episode(
        h.graphiti,
        h.store,
        h.config,
        unit_id='u2',
        group_id=GROUP,
        name='episode',
        episode_body='content',
        source_description='test',
        reference_time=T0,
    )

    assert status(h, stopped.seq) == STATUS_FAILED
    assert unit_2.seq == stopped.seq + 1
    # The episode node is visible with an empty `entity_edges` list.
    assert h.graph.episodes[unit_2.episodes[0].uuid].entity_edges == []
    (span,) = h.tracer.named('ingest.admit')
    assert span.attributes['ingest.seq'] == unit_2.seq


async def test_commit_waits_for_the_earlier_unit(monkeypatch):
    h = Harness(monkeypatch)
    unit_1 = await h.admit('u1')
    unit_2 = await h.admit('u2')
    change_set_1 = await h.analyze(unit_1, create_nodes=[make_node('a1', 'Acme')])
    change_set_2 = await h.analyze(unit_2, create_nodes=[make_node('a2', 'Acme')])
    assert isinstance(await h.store.begin_commit(GROUP, unit_2.seq, 60.0), CommitWait)

    commit_2 = asyncio.create_task(h.commit(unit_2, change_set_2))
    await asyncio.sleep(0.02)
    assert not commit_2.done()
    assert h.graph.nodes == {}

    await h.commit(unit_1, change_set_1)
    result_2 = await commit_2

    # The commits are in sequence order, so unit 2 finds the node of unit 1.
    assert result_2.change_set.uuid_map == {'a2': 'a1'}
    assert result_2.report['turn_wait_ms'] > 0


async def test_cancellation_before_the_final_change_set_releases_the_position(monkeypatch):
    h = Harness(monkeypatch)
    unit_1 = await h.admit('u1')
    unit_2 = await h.admit('u2')
    unit_3 = await h.admit('u3')
    change_set_1 = await h.analyze(unit_1, create_nodes=[make_node('a1', 'Acme')])
    change_set_2 = await h.analyze(unit_2, create_nodes=[make_node('a2', 'Bolt', Y)])
    change_set_3 = await h.analyze(unit_3)

    commit_2 = asyncio.create_task(h.commit(unit_2, change_set_2))
    await asyncio.sleep(0.02)
    commit_2.cancel()
    with pytest.raises(asyncio.CancelledError):
        await commit_2

    assert status(h, unit_2.seq) == STATUS_FAILED
    await h.commit(unit_1, change_set_1)
    await h.commit(unit_3, change_set_3)
    assert set(h.graph.nodes) == {'a1'}


# --- failures after the final change set is stored --------------------------------


async def test_failed_write_is_applied_again_with_no_model_call(monkeypatch):
    h = Harness(monkeypatch)
    h.graph.fail_bulk_writes = 2
    h.graph.partial_bulk_write = True
    unit, result = await h.run('u1', create_nodes=[make_node('a1', 'Acme')])

    assert result.report['apply.attempts'] == 3
    assert h.graph.log.count('store:save_final') == 1
    assert h.graph.log.count('write:bulk') == 3
    assert status(h, unit.seq) == STATUS_COMMITTED
    assert set(h.graph.nodes) == {'a1'}
    h.llm.assert_not_awaited()


async def test_all_apply_attempts_fail(monkeypatch):
    h = Harness(monkeypatch, max_apply_attempts=3)
    h.graph.fail_bulk_writes = 3
    unit_1 = await h.admit('u1')
    change_set_1 = await h.analyze(unit_1, create_nodes=[make_node('a1', 'Acme')])

    with pytest.raises(RuntimeError, match='bulk write failed'):
        await h.commit(unit_1, change_set_1)

    assert h.graph.log.count('write:bulk') == 3
    stored = h.store._groups[GROUP].units[unit_1.seq]
    assert stored.status == STATUS_APPLY_FAILED
    # The final change set stays for the delta of later units.
    assert stored.payload is not None
    assert h.store._groups[GROUP].watermark == unit_1.seq
    # No attempt writes the change set again, and the next unit commits.
    _, result_2 = await h.run('u2', create_nodes=[make_node('b2', 'Bo', Y)])
    assert set(h.graph.nodes) == {'b2'}
    assert result_2.report['recovered.count'] == 0


async def stop_after_final(h: Harness, unit_id: str, **fields):
    """Make a unit that stored its final change set and then stopped."""
    unit = await h.admit(unit_id)
    change_set = await h.analyze(unit, **fields)
    turn = await h.store.begin_commit(GROUP, unit.seq, h.config.lease_seconds)
    await h.store.save_final(GROUP, unit.seq, turn.fence, change_set.to_bytes())
    return unit, turn


async def test_next_unit_recovers_a_unit_that_stopped_after_the_final_change_set(monkeypatch):
    clock = FakeClock()
    h = Harness(monkeypatch, clock=clock)
    unit_1, _ = await stop_after_final(h, 'u1', create_nodes=[make_node('a1', 'Acme')])
    unit_2 = await h.admit('u2')
    change_set_2 = await h.analyze(unit_2, create_nodes=[make_node('a2', 'Acme')])
    lease_2 = UnitLease(h.store, h.config, GROUP, unit_2.seq, clock=clock)

    # Before the deadline of unit 1, unit 2 must wait.
    assert isinstance(await h.store.begin_commit(GROUP, unit_2.seq, 60.0), CommitWait)
    # Unit 2 renews its deadline. The deadline of unit 1 then expires.
    clock.advance(h.config.lease_seconds - 1)
    await lease_2.heartbeat()
    clock.advance(2)

    result_2 = await h.commit(unit_2, change_set_2, lease=lease_2)

    assert status(h, unit_1.seq) == STATUS_COMMITTED
    assert result_2.report['recovered.count'] == 1
    # Unit 2 commits after the recovery, so it finds the node of unit 1.
    assert result_2.change_set.uuid_map == {'a2': 'a1'}
    assert set(h.graph.nodes) == {'a1'}
    # The nodes of the recovered unit get no enrichment from that unit.
    assert await h.store.ready_enrich(GROUP, unit_2.seq) == ['a1']
    assert h.store._groups[GROUP].enrich.get(unit_1.seq) is None
    h.llm.assert_not_awaited()


async def test_recovery_that_fails_marks_the_stopped_unit_apply_failed(monkeypatch):
    clock = FakeClock()
    h = Harness(monkeypatch, clock=clock, max_apply_attempts=2)
    unit_1, _ = await stop_after_final(h, 'u1', create_nodes=[make_node('a1', 'Acme')])
    unit_2 = await h.admit('u2')
    change_set_2 = await h.analyze(unit_2, create_nodes=[make_node('b2', 'Bo', Y)])
    lease_2 = UnitLease(h.store, h.config, GROUP, unit_2.seq, clock=clock)
    # Unit 2 renews its deadline. The deadline of unit 1 then expires.
    clock.advance(h.config.lease_seconds - 1)
    await lease_2.heartbeat()
    clock.advance(2)
    h.graph.fail_bulk_writes = 2

    result_2 = await h.commit(unit_2, change_set_2, lease=lease_2)

    assert status(h, unit_1.seq) == STATUS_APPLY_FAILED
    assert status(h, unit_2.seq) == STATUS_COMMITTED
    assert result_2.report['recovered.count'] == 0
    assert set(h.graph.nodes) == {'b2'}


async def test_stale_attempt_cannot_record_the_commit(monkeypatch):
    clock = FakeClock()
    h = Harness(monkeypatch, clock=clock)
    unit, stale_turn = await stop_after_final(h, 'u1', create_nodes=[make_node('a1', 'Acme')])
    clock.advance(h.config.lease_seconds + 1)

    # A second attempt of the same unit gets the turn and the stored final change set.
    lease = UnitLease(h.store, h.config, GROUP, unit.seq, clock=clock)
    final = ChangeSet.from_bytes(h.store._groups[GROUP].units[unit.seq].payload)
    result = await h.commit(unit, final, lease=lease)
    assert result.report['retry'] == 1

    with pytest.raises(StaleCommitError):
        await h.store.finish_commit(GROUP, unit.seq, stale_turn.fence, ['a1'], 0.0)
    with pytest.raises(StaleCommitError):
        await h.store.save_final(GROUP, unit.seq, stale_turn.fence, b'x')


async def test_second_attempt_waits_for_the_deadline_of_the_applying_attempt(monkeypatch):
    clock = FakeClock()
    h = Harness(monkeypatch, clock=clock)
    unit, first_turn = await stop_after_final(h, 'u1', create_nodes=[make_node('a1', 'Acme')])
    final = ChangeSet.from_bytes(h.store._groups[GROUP].units[unit.seq].payload)
    first_deadline = h.store._groups[GROUP].units[unit.seq].deadline
    waits: list[float] = []

    async def wait() -> None:
        # The second attempt writes nothing while the first attempt has the turn. Its
        # heartbeat does not move the deadline of the unit.
        assert 'write:bulk' not in h.graph.log
        assert h.store._groups[GROUP].units[unit.seq].deadline == first_deadline
        assert len(waits) < 10
        waits.append(clock())
        clock.advance(h.config.lease_seconds / 2)

    result = await h.commit(unit, final, wait=wait)

    assert len(waits) == 2
    assert result.report['retry'] == 1
    assert set(h.graph.nodes) == {'a1'}
    with pytest.raises(StaleCommitError):
        await h.store.heartbeat(GROUP, unit.seq, h.config.lease_seconds, first_turn.fence)


async def test_second_analysis_of_a_unit_keeps_the_stored_final_change_set(monkeypatch):
    clock = FakeClock()
    h = Harness(monkeypatch, clock=clock)
    unit, _ = await stop_after_final(h, 'u1', create_nodes=[make_node('a1', 'Acme')])

    # A retry of the unit runs the analysis again. The result has different records.
    second = await h.analyze(unit, create_nodes=[make_node('other', 'Acme')])
    clock.advance(h.config.lease_seconds + 1)
    result = await h.commit(unit, second)

    assert result.report['retry'] == 1
    assert set(h.graph.nodes) == {'a1'}
    assert status(h, unit.seq) == STATUS_COMMITTED
    h.llm.assert_not_awaited()


async def test_lease_heartbeat_is_stale_after_a_different_attempt_gets_the_turn(monkeypatch):
    clock = FakeClock()
    h = Harness(monkeypatch, clock=clock)
    unit, first_turn = await stop_after_final(h, 'u1', create_nodes=[make_node('a1', 'Acme')])
    first_lease = UnitLease(h.store, h.config, GROUP, unit.seq, clock=clock)
    first_lease.fence = first_turn.fence
    await first_lease.heartbeat()
    budget = first_lease.write_budget()

    clock.advance(h.config.lease_seconds + 1)
    retry = await h.store.begin_commit(GROUP, unit.seq, h.config.lease_seconds)
    assert retry.fence == first_turn.fence + 1

    # The first attempt cannot renew its local deadline, and it cannot write.
    with pytest.raises(StaleCommitError):
        await first_lease.heartbeat()
    assert budget > 0
    with pytest.raises(UnitExpiredError):
        first_lease.write_budget()


async def test_expired_unit_stops_at_the_next_store_call(monkeypatch):
    clock = FakeClock()
    h = Harness(monkeypatch, clock=clock)
    unit = await h.admit('u1')
    change_set = await h.analyze(unit, create_nodes=[make_node('a1', 'Acme')])
    clock.advance(h.config.lease_seconds + 1)

    with pytest.raises(UnitExpiredError):
        await h.commit(unit, change_set)

    assert h.graph.nodes == {}
    assert 'write:bulk' not in h.graph.log


# --- Enrich ------------------------------------------------------------------------


async def test_enrichment_of_one_node_runs_in_sequence_order(monkeypatch):
    h = Harness(monkeypatch)
    h.graph.add(make_node('s1', 'Shared'))
    unit_1, result_1 = await h.run('u1', node_uuids=['s1'], create_nodes=[])
    unit_2, result_2 = await h.run(
        'u2', create_nodes=[make_node('b2', 'Bo', Y)], node_uuids=['s1', 'b2']
    )
    # The shared node waits for unit 1. The other node of unit 2 does not wait.
    assert await h.store.ready_enrich(GROUP, unit_2.seq) == ['b2']

    enrich_2 = asyncio.create_task(h.enrich(unit_2, result_2))
    await asyncio.sleep(0.02)
    assert not enrich_2.done()
    assert h.graph.nodes['b2'].summary == '+ep-u2'
    assert h.graph.nodes['s1'].summary == ''

    nodes_1 = await h.enrich(unit_1, result_1)
    nodes_2 = await enrich_2

    assert h.graph.nodes['s1'].summary == '+ep-u1+ep-u2'
    assert [node.uuid for node in nodes_1] == ['s1']
    assert [node.uuid for node in nodes_2] == ['s1', 'b2']
    assert h.store._groups[GROUP].enrich == {}
    span = h.tracer.named('ingest.enrich')[0]
    assert span.attributes['batch.count'] == 2
    assert span.attributes['node.count'] == 2
    assert span.attributes['wait_ms'] > 0


async def test_enrichment_applies_the_patch_to_the_current_node(monkeypatch):
    h = Harness(monkeypatch)
    h.graph.add(make_node('s1', 'Acme', attributes={'city': 'Oslo'}))
    unit, result = await h.run(
        'u1',
        node_uuids=['s1'],
        operations=[
            PatchNode(
                node_uuid='s1', labels=['Entity', 'Company'], attributes={'domain': 'acme.com'}
            )
        ],
    )
    # A different writer changes the node after the commit.
    h.graph.nodes['s1'].attributes['city'] = 'Bergen'

    await h.enrich(unit, result)

    node = h.graph.nodes['s1']
    assert node.labels == ['Entity', 'Company']
    assert node.attributes == {'city': 'Bergen', 'domain': 'acme.com'}
    assert h.graph.write_timeouts[-1] is not None


async def test_enrichment_failure_keeps_the_node_values_and_does_not_raise(monkeypatch):
    h = Harness(monkeypatch)
    h.extract_attributes.side_effect = RuntimeError('model')
    unit, result = await h.run('u1', create_nodes=[make_node('a1', 'Acme', summary='before')])

    nodes = await h.enrich(unit, result)

    assert [node.uuid for node in nodes] == ['a1']
    assert h.graph.nodes['a1'].summary == 'before'
    assert h.store._groups[GROUP].enrich == {}
    (span,) = h.tracer.named('ingest.enrich')
    assert span.attributes['node.failed'] == 1


async def test_enrichment_hook_selects_the_nodes_to_write(monkeypatch):
    hook_units: list[str] = []

    async def keep_first(nodes, unit):
        hook_units.append(unit.unit_id)
        return [node for node in nodes if node.uuid == 'a1']

    h = Harness(monkeypatch, before_enrichment_write=keep_first)
    unit, result = await h.run(
        'u1', create_nodes=[make_node('a1', 'Acme'), make_node('b1', 'Bo', Y)]
    )

    nodes = await h.enrich(unit, result)

    assert hook_units == ['u1']
    assert [node.uuid for node in nodes] == ['a1']
    assert h.graph.nodes['a1'].summary == '+ep-u1'
    assert h.graph.nodes['b1'].summary == ''


async def test_enrichment_does_not_create_again_a_node_that_a_writer_deleted(monkeypatch):
    h = Harness(monkeypatch)
    unit, result = await h.run(
        'u1', create_nodes=[make_node('a1', 'Acme'), make_node('b1', 'Bo', Y)]
    )

    async def delete_node(clients, nodes, episode=None, *args, **kwargs):
        # A writer outside the sequence deletes a node between the read and the write.
        del h.graph.nodes['b1']
        return nodes

    h.extract_attributes.side_effect = delete_node
    await h.enrich(unit, result)

    assert set(h.graph.nodes) == {'a1'}
    assert 'write:node_update' in h.graph.log


async def test_enrichment_write_error_releases_the_enrichment_order(monkeypatch):
    h = Harness(monkeypatch)
    h.graph.add(make_node('s1', 'Shared'))
    unit_1, result_1 = await h.run('u1', node_uuids=['s1'])
    unit_2, result_2 = await h.run('u2', node_uuids=['s1'])
    h.graph.fail_node_updates = 1

    with pytest.raises(RuntimeError, match='node update failed'):
        await h.enrich(unit_1, result_1)

    # Unit 2 does not wait for the entries of unit 1.
    nodes_2 = await h.enrich(unit_2, result_2)
    assert [node.uuid for node in nodes_2] == ['s1']
    assert h.graph.nodes['s1'].summary == '+ep-u2'
