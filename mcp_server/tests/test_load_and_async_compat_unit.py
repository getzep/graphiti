import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))

from test_stress_load import LoadTestConfig, LoadTester


class _RecordingSession:
    def __init__(self, operation_name: str):
        self.operation_name = operation_name
        self.calls: list[tuple[str, dict]] = []

    async def call_tool(self, operation: str, args: dict):
        self.calls.append((operation, args))
        return SimpleNamespace(content=[])


@pytest.mark.asyncio
async def test_load_tester_uses_current_tool_names_and_parameters(monkeypatch):
    tester = LoadTester(
        LoadTestConfig(
            num_clients=1,
            operations_per_client=1,
            ramp_up_time=0,
            test_duration=5,
            think_time=0,
        )
    )
    tester.start_time = 0
    session = _RecordingSession('search_nodes')

    monkeypatch.setattr(
        'test_stress_load.random.choice',
        lambda options: 'search_nodes',
    )

    await tester.run_client_workload(0, session, 'group-1')

    assert session.calls == [('search_nodes', {'query': 'architecture', 'group_ids': ['group-1'], 'max_nodes': 10})] or session.calls[0][0] == 'search_nodes'
    operation, args = session.calls[0]
    assert operation == 'search_nodes'
    assert args['group_ids'] == ['group-1']
    assert args['max_nodes'] == 10


@pytest.mark.asyncio
async def test_load_tester_get_episodes_uses_max_episodes(monkeypatch):
    tester = LoadTester(
        LoadTestConfig(
            num_clients=1,
            operations_per_client=1,
            ramp_up_time=0,
            test_duration=5,
            think_time=0,
        )
    )
    tester.start_time = 0
    session = _RecordingSession('get_episodes')

    monkeypatch.setattr(
        'test_stress_load.random.choice',
        lambda options: 'get_episodes',
    )

    await tester.run_client_workload(0, session, 'group-1')

    operation, args = session.calls[0]
    assert operation == 'get_episodes'
    assert args['group_ids'] == ['group-1']
    assert args['max_episodes'] == 10
