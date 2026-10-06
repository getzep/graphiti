from datetime import datetime, timezone
from types import SimpleNamespace

from graphiti_core.graphiti import (
    MAX_SAGA_EPISODES_FOR_SUMMARY,
    SAGA_SUMMARY_MAX_EPISODE_CHARS,
    SAGA_SUMMARY_MAX_TOKENS,
    Graphiti,
    select_saga_summary_episodes,
)
from graphiti_core.nodes import SagaNode
from graphiti_core.utils.text_utils import SAGA_SUMMARY_MAX_CHARS


def _dt(minute: int) -> datetime:
    return datetime(2026, 1, 1, 0, minute, tzinfo=timezone.utc)


def test_saga_summary_budget_constants():
    assert MAX_SAGA_EPISODES_FOR_SUMMARY == 20
    assert SAGA_SUMMARY_MAX_EPISODE_CHARS == 100_000
    assert SAGA_SUMMARY_MAX_TOKENS == 8192
    assert SAGA_SUMMARY_MAX_CHARS == 4000


def test_saga_summary_episode_selection_respects_byte_budget():
    episodes = [
        ('a' * 40, None, _dt(1)),
        ('b' * 40, None, _dt(2)),
        ('c' * 40, None, _dt(3)),
    ]

    selected, selected_bytes, dropped = select_saga_summary_episodes(episodes, max_episode_chars=90)

    assert [content for content, _, _ in selected] == ['a' * 40, 'b' * 40]
    assert [created_at for _, _, created_at in selected] == [_dt(1), _dt(2)]
    assert selected_bytes == 80
    assert dropped is True


def test_saga_summary_episode_selection_counts_characters_not_utf8_bytes():
    episodes = [
        ('é' * 10, None, _dt(1)),
        ('a' * 10, None, _dt(2)),
    ]

    selected, selected_bytes, dropped = select_saga_summary_episodes(episodes, max_episode_chars=15)

    assert [content for content, _, _ in selected] == ['é' * 10]
    assert selected_bytes == len(('é' * 10).encode('utf-8'))
    assert dropped is True


def test_saga_summary_episode_selection_truncates_first_oversized_episode():
    selected, selected_bytes, dropped = select_saga_summary_episodes(
        [('é' * 10, None, _dt(1))], max_episode_chars=7
    )

    assert [content for content, _, _ in selected] == ['é' * 7]
    assert selected_bytes == len(('é' * 7).encode('utf-8'))
    assert dropped is True


class _FakeGraphOperations:
    def __init__(self, episodes=None):
        self.episodes = episodes or [
            ('a' * 40, _dt(1), _dt(11)),
            ('b' * 40, _dt(2), _dt(12)),
            ('c' * 40, _dt(3), _dt(13)),
        ]
        self.saved_saga = None

    async def saga_node_get_by_uuid(self, _cls, _driver, uuid: str):
        return SagaNode(uuid=uuid, name='test saga', group_id='graph-1', created_at=_dt(0))

    async def saga_get_episode_contents(self, _driver, _saga_uuid, since=None, limit=200):
        assert since is None
        assert limit == MAX_SAGA_EPISODES_FOR_SUMMARY
        return self.episodes

    async def saga_node_save(self, saga, _driver):
        self.saved_saga = saga


class _FakeLLMClient:
    def __init__(self):
        self.messages = None
        self.max_tokens = None

    async def generate_response(
        self, messages, response_model=None, prompt_name=None, max_tokens=None, **_kwargs
    ):
        assert prompt_name == 'summarize_sagas.summarize_saga'
        self.messages = messages
        self.max_tokens = max_tokens
        return {'summary': 'updated summary'}


class _FailingLLMClient:
    async def generate_response(self, _messages, response_model=None, prompt_name=None, **_kwargs):
        raise AssertionError('LLM should not be called for this case')


async def test_saga_summary_cursor_advances_only_to_latest_selected_episode(monkeypatch):
    monkeypatch.setattr('graphiti_core.graphiti.SAGA_SUMMARY_MAX_EPISODE_CHARS', 90)
    operations = _FakeGraphOperations()
    llm = _FakeLLMClient()
    graphiti = Graphiti.__new__(Graphiti)
    graphiti.driver = SimpleNamespace(graph_operations_interface=operations)
    graphiti.llm_client = llm

    saga = await graphiti.summarize_saga('saga-1')

    assert saga.summary == 'updated summary'
    assert saga.last_summarized_at == _dt(12)
    assert saga.last_summarized_episode_valid_at == _dt(2)
    assert saga._summary_episodes_fetched == 3
    assert saga._summary_episodes_selected == 2
    assert saga._summary_episodes_skipped == 1
    assert saga._summary_selected_episode_bytes == 80
    assert operations.saved_saga is saga
    assert llm.max_tokens == SAGA_SUMMARY_MAX_TOKENS


async def test_saga_summary_watermark_advances_for_partial_fetch_window(monkeypatch):
    """A 20-episode fetch that only fits five episodes must advance
    last_summarized_at to the fifth episode's created_at so the rest remain
    eligible on the next run."""
    monkeypatch.setattr('graphiti_core.graphiti.SAGA_SUMMARY_MAX_EPISODE_CHARS', 5_000)
    episodes = [('e' * 1_000, _dt(i), _dt(10 + i)) for i in range(1, 21)]
    operations = _FakeGraphOperations(episodes=episodes)
    llm = _FakeLLMClient()
    graphiti = Graphiti.__new__(Graphiti)
    graphiti.driver = SimpleNamespace(graph_operations_interface=operations)
    graphiti.llm_client = llm

    saga = await graphiti.summarize_saga('saga-1')

    assert saga._summary_episodes_fetched == 20
    assert saga._summary_episodes_selected == 5
    assert saga._summary_episodes_skipped == 15
    assert saga.last_summarized_at == _dt(15)
    assert saga.last_summarized_episode_valid_at == _dt(5)


class _LongSummaryLLMClient(_FakeLLMClient):
    async def generate_response(
        self, messages, response_model=None, prompt_name=None, max_tokens=None, **_kwargs
    ):
        await super().generate_response(
            messages, response_model=response_model, prompt_name=prompt_name, max_tokens=max_tokens
        )
        return {
            'summary': ('Kept sentence. ' + ('x' * (SAGA_SUMMARY_MAX_CHARS + 50)) + ' Dropped.')
        }


async def test_saga_summary_truncates_stored_text_to_8k_chars():
    operations = _FakeGraphOperations()
    llm = _LongSummaryLLMClient()
    graphiti = Graphiti.__new__(Graphiti)
    graphiti.driver = SimpleNamespace(graph_operations_interface=operations)
    graphiti.llm_client = llm

    saga = await graphiti.summarize_saga('saga-1')

    assert len(saga.summary) <= SAGA_SUMMARY_MAX_CHARS
    assert saga.summary.startswith('Kept sentence.')
    assert 'Dropped.' not in saga.summary
    assert llm.max_tokens == SAGA_SUMMARY_MAX_TOKENS


async def test_saga_summary_cursor_follows_created_at_not_valid_at(monkeypatch):
    monkeypatch.setattr('graphiti_core.graphiti.SAGA_SUMMARY_MAX_EPISODE_CHARS', 50)
    operations = _FakeGraphOperations(
        episodes=[
            ('regular earlier ingestion', _dt(20), _dt(9)),
            ('backfill newer ingestion but older valid_at', _dt(1), _dt(10)),
        ]
    )
    graphiti = Graphiti.__new__(Graphiti)
    graphiti.driver = SimpleNamespace(graph_operations_interface=operations)
    graphiti.llm_client = _FakeLLMClient()

    saga = await graphiti.summarize_saga('saga-1')

    assert saga.summary == 'updated summary'
    assert saga.last_summarized_at == _dt(9)
    assert saga.last_summarized_episode_valid_at == _dt(20)
    assert operations.saved_saga is saga


async def test_saga_summary_prompt_orders_selected_episodes_by_valid_at(monkeypatch):
    monkeypatch.setattr('graphiti_core.graphiti.SAGA_SUMMARY_MAX_EPISODE_CHARS', 200)
    operations = _FakeGraphOperations(
        episodes=[
            ('regular earlier ingestion', _dt(20), _dt(9)),
            ('backfill newer ingestion but older valid_at', _dt(1), _dt(10)),
        ]
    )
    graphiti = Graphiti.__new__(Graphiti)
    graphiti.driver = SimpleNamespace(graph_operations_interface=operations)
    graphiti.llm_client = _FakeLLMClient()

    await graphiti.summarize_saga('saga-1')

    user_prompt = graphiti.llm_client.messages[1].content
    assert user_prompt.index('backfill newer ingestion') < user_prompt.index(
        'regular earlier ingestion'
    )


async def test_saga_summary_truncates_oversized_first_episode_and_advances_cursor(
    monkeypatch,
):
    monkeypatch.setattr('graphiti_core.graphiti.SAGA_SUMMARY_MAX_EPISODE_CHARS', 10)
    operations = _FakeGraphOperations()
    llm = _FakeLLMClient()
    graphiti = Graphiti.__new__(Graphiti)
    graphiti.driver = SimpleNamespace(graph_operations_interface=operations)
    graphiti.llm_client = llm

    saga = await graphiti.summarize_saga('saga-1')

    assert saga.summary == 'updated summary'
    assert saga.last_summarized_at == _dt(11)
    assert saga._summary_episodes_fetched == 3
    assert saga._summary_episodes_selected == 1
    assert saga._summary_episodes_skipped == 2
    assert saga._summary_selected_episode_bytes == 10
    assert llm.max_tokens == SAGA_SUMMARY_MAX_TOKENS
    assert operations.saved_saga is saga


class _RebuildFakeGraphOperations(_FakeGraphOperations):
    """A saga that HAS been summarized before: carries a prior summary and
    watermarks, and records the `since` filter the fetch was given."""

    def __init__(self, episodes=None):
        super().__init__(episodes)
        self.requested_since = 'unset'

    async def saga_node_get_by_uuid(self, _cls, _driver, uuid: str):
        saga = SagaNode(uuid=uuid, name='test saga', group_id='graph-1', created_at=_dt(0))
        saga.summary = 'OLD SUMMARY CONTAINING DELETED EPISODE CONTENT'
        saga.last_summarized_at = _dt(20)
        saga.last_summarized_episode_valid_at = _dt(9)
        return saga

    async def saga_get_episode_contents(self, _driver, _saga_uuid, since=None, limit=200):
        self.requested_since = since
        return self.episodes


class _CapturingLLMClient:
    def __init__(self):
        self.messages = None

    async def generate_response(
        self, messages, response_model=None, prompt_name=None, max_tokens=None, **_kwargs
    ):
        self.messages = messages
        self.max_tokens = max_tokens
        return {'summary': 'rebuilt summary'}


async def test_saga_summary_rebuild_ignores_watermark_and_discards_old_text():
    """abac spec-10 §6.2: a rebuild regenerates from scratch -- the watermark
    filter is bypassed (all surviving episodes fetched) and the prior summary
    text (which carries deleted-episode content) is not fed to the prompt."""
    operations = _RebuildFakeGraphOperations()
    llm = _CapturingLLMClient()
    graphiti = Graphiti.__new__(Graphiti)
    graphiti.driver = SimpleNamespace(graph_operations_interface=operations)
    graphiti.llm_client = llm

    saga = await graphiti.summarize_saga('saga-1', rebuild=True)

    assert operations.requested_since is None, 'rebuild must fetch ALL surviving episodes'
    prompt_text = ' '.join(str(m) for m in llm.messages)
    assert 'OLD SUMMARY CONTAINING DELETED EPISODE CONTENT' not in prompt_text, (
        'rebuild must not feed the prior summary back into the prompt'
    )
    assert saga.summary == 'rebuilt summary'
    # The temporal watermark reflects the surviving episodes even though it
    # regresses from the pre-rebuild value.
    assert saga.last_summarized_episode_valid_at == _dt(3)
    assert operations.saved_saga is saga


async def test_saga_summary_rebuild_with_no_survivors_clears_summary():
    """A rebuild over zero surviving episodes must not invent content: the
    summary is cleared without an LLM call."""
    operations = _RebuildFakeGraphOperations(episodes=[])
    operations.episodes = []
    graphiti = Graphiti.__new__(Graphiti)
    graphiti.driver = SimpleNamespace(graph_operations_interface=operations)
    graphiti.llm_client = _FailingLLMClient()

    saga = await graphiti.summarize_saga('saga-1', rebuild=True)

    assert saga.summary == ''
    assert saga.last_summarized_episode_valid_at is None
    # content policies spec-1 §7.2: an episode that is not eligible yet (a
    # pending policy state) is behind no watermark, so the next incremental
    # run reads it.
    assert saga.last_summarized_at is None
    assert operations.saved_saga is saga


async def test_saga_summary_prompt_stays_under_context_cap(monkeypatch):
    monkeypatch.setattr('graphiti_core.graphiti.SAGA_SUMMARY_MAX_PROMPT_BYTES', 12_000)
    monkeypatch.setattr('graphiti_core.graphiti.SAGA_SUMMARY_MAX_EPISODE_CHARS', 12_000)
    operations = _FakeGraphOperations(episodes=[('x' * 4_000, _dt(i), _dt(i)) for i in range(1, 8)])
    llm = _FakeLLMClient()
    graphiti = Graphiti.__new__(Graphiti)
    graphiti.driver = SimpleNamespace(graph_operations_interface=operations)
    graphiti.llm_client = llm

    saga = await graphiti.summarize_saga('saga-1')

    prompt_bytes = sum(len(message.content.encode('utf-8')) for message in llm.messages)
    assert prompt_bytes <= 12_000
    assert saga._summary_episodes_selected < 7
    assert saga._summary_episodes_skipped > 0
    assert saga.last_summarized_at == _dt(saga._summary_episodes_selected)
    assert llm.max_tokens == SAGA_SUMMARY_MAX_TOKENS


async def test_saga_summary_before_publish_runs_before_save_and_can_stop_it():
    """The publication callback runs after generation and before the save. An
    exception from the callback leaves the saga unsaved."""
    operations = _FakeGraphOperations()
    graphiti = Graphiti.__new__(Graphiti)
    graphiti.driver = SimpleNamespace(graph_operations_interface=operations)
    graphiti.llm_client = _FakeLLMClient()
    seen: list[str] = []

    async def before_publish(summary: str) -> str:
        assert operations.saved_saga is None
        seen.append(summary)
        return summary + ' (checked)'

    saga = await graphiti.summarize_saga('saga-1', before_publish=before_publish)

    assert seen == ['updated summary']
    assert saga.summary == 'updated summary (checked)'
    assert operations.saved_saga is saga

    class _Stop(Exception):
        pass

    async def stop(_summary: str) -> str:
        raise _Stop()

    operations = _FakeGraphOperations()
    graphiti.driver = SimpleNamespace(graph_operations_interface=operations)
    try:
        await graphiti.summarize_saga('saga-1', before_publish=stop)
    except _Stop:
        pass
    else:
        raise AssertionError('the callback must stop the publication')
    assert operations.saved_saga is None
