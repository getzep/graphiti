#!/usr/bin/env python3
"""Unit tests for rate-limit retries on provider calls and the episode dead-letter file."""

import asyncio
import json
import sys
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import pytest

# Add the src directory to the path (mirrors the other unit tests)
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from graphiti_core.embedder.client import EmbedderClient
from graphiti_core.llm_client import LLMClient
from graphiti_core.llm_client.errors import RateLimitError
from graphiti_core.nodes import EpisodeType
from graphiti_core.prompts.models import Message

from services.queue_service import QueueService
from services.rate_limit_retry import (
    RateLimitRetryEmbedder,
    RateLimitRetryPolicy,
    add_llm_rate_limit_retry,
    is_rate_limit_error,
    retry_on_rate_limit,
)


class ClientError(Exception):
    """Mirrors google.genai.errors.ClientError (google-genai is an optional extra)."""

    def __init__(self, code: int, status: str, message: str):
        super().__init__(f'{code} {status}. {message}')
        self.code = code
        self.status = status


def google_resource_exhausted() -> ClientError:
    """The error GeminiEmbedder surfaces when Gemini returns HTTP 429."""
    return ClientError(429, 'RESOURCE_EXHAUSTED', 'Resource has been exhausted (e.g. check quota).')


class RecordingSleep:
    """Stand-in for asyncio.sleep that records delays instead of waiting."""

    def __init__(self):
        self.delays: list[float] = []

    async def __call__(self, delay: float) -> None:
        self.delays.append(delay)


class FlakyEmbedder(EmbedderClient):
    """Embedder that raises the given errors before returning a vector."""

    def __init__(self, errors: list[Exception]):
        self.errors = list(errors)
        self.calls = 0
        self.config = MagicMock(embedding_dim=3)

    async def create(self, input_data):
        self.calls += 1
        if self.errors:
            raise self.errors.pop(0)
        return [0.1, 0.2, 0.3]

    async def create_batch(self, input_data_list):
        self.calls += 1
        if self.errors:
            raise self.errors.pop(0)
        return [[0.1, 0.2, 0.3] for _ in input_data_list]


class TestRateLimitRetryPolicy:
    def test_default_schedule_is_10_30_90(self):
        assert RateLimitRetryPolicy().delays() == [10.0, 30.0, 90.0]

    def test_from_env(self, monkeypatch):
        monkeypatch.setenv('RATE_LIMIT_MAX_RETRIES', '4')
        monkeypatch.setenv('RATE_LIMIT_RETRY_BASE_DELAY', '2')
        monkeypatch.setenv('RATE_LIMIT_RETRY_MULTIPLIER', '2')
        monkeypatch.setenv('RATE_LIMIT_RETRY_MAX_DELAY', '10')
        assert RateLimitRetryPolicy.from_env().delays() == [2.0, 4.0, 8.0, 10.0]

    def test_from_env_ignores_invalid_values(self, monkeypatch):
        monkeypatch.setenv('RATE_LIMIT_MAX_RETRIES', 'lots')
        monkeypatch.setenv('RATE_LIMIT_RETRY_BASE_DELAY', '-5')
        assert RateLimitRetryPolicy.from_env() == RateLimitRetryPolicy()

    def test_zero_retries_disables_retrying(self, monkeypatch):
        monkeypatch.setenv('RATE_LIMIT_MAX_RETRIES', '0')
        assert RateLimitRetryPolicy.from_env().delays() == []


class TestIsRateLimitError:
    @pytest.mark.parametrize(
        'exc',
        [
            google_resource_exhausted(),
            RateLimitError('Rate limit exceeded. Please try again later.'),
            Exception('429 RESOURCE_EXHAUSTED. quota exceeded'),
            Exception('Error code: 429 - rate limit reached for requests'),
        ],
    )
    def test_detects_rate_limits(self, exc):
        assert is_rate_limit_error(exc)

    def test_detects_real_google_genai_error(self):
        genai_errors = pytest.importorskip('google.genai.errors')
        exc = genai_errors.ClientError(
            429,
            {
                'error': {
                    'code': 429,
                    'message': 'Resource has been exhausted (e.g. check quota).',
                    'status': 'RESOURCE_EXHAUSTED',
                }
            },
        )
        assert is_rate_limit_error(exc)

    def test_detects_provider_sdk_rate_limit_class(self):
        class RateLimitError(Exception):  # e.g. anthropic.RateLimitError / openai.RateLimitError
            pass

        assert is_rate_limit_error(RateLimitError('slow down'))

    def test_detects_http_429_status(self):
        exc = Exception('HTTP error')
        exc.response = MagicMock(status_code=429)  # type: ignore[attr-defined]
        assert is_rate_limit_error(exc)

    def test_detects_wrapped_cause(self):
        try:
            try:
                raise google_resource_exhausted()
            except Exception as inner:
                raise RuntimeError('embedding failed') from inner
        except RuntimeError as outer:
            assert is_rate_limit_error(outer)

    @pytest.mark.parametrize(
        'exc',
        [
            ValueError('No embeddings returned from Gemini API'),
            ClientError(400, 'INVALID_ARGUMENT', 'bad request'),
            Exception('episode 429 not found'),
        ],
    )
    def test_ignores_other_errors(self, exc):
        assert not is_rate_limit_error(exc)


class TestRetryOnRateLimit:
    async def test_retries_then_succeeds(self):
        sleep = RecordingSleep()
        func = AsyncMock(side_effect=[google_resource_exhausted(), RateLimitError(), 'ok'])

        result = await retry_on_rate_limit(
            func, policy=RateLimitRetryPolicy(), operation='test', sleep=sleep
        )

        assert result == 'ok'
        assert func.await_count == 3
        assert sleep.delays == [10.0, 30.0]

    async def test_gives_up_after_max_retries(self):
        sleep = RecordingSleep()
        func = AsyncMock(side_effect=google_resource_exhausted())

        with pytest.raises(ClientError):
            await retry_on_rate_limit(
                func, policy=RateLimitRetryPolicy(), operation='test', sleep=sleep
            )

        assert func.await_count == 4  # first attempt + 3 retries
        assert sleep.delays == [10.0, 30.0, 90.0]

    async def test_does_not_retry_other_errors(self):
        sleep = RecordingSleep()
        func = AsyncMock(side_effect=ValueError('bad input'))

        with pytest.raises(ValueError):
            await retry_on_rate_limit(
                func, policy=RateLimitRetryPolicy(), operation='test', sleep=sleep
            )

        assert func.await_count == 1
        assert sleep.delays == []


class TestRateLimitRetryEmbedder:
    async def test_create_retries_google_429(self):
        inner = FlakyEmbedder([google_resource_exhausted()])
        sleep = RecordingSleep()
        embedder = RateLimitRetryEmbedder(inner, RateLimitRetryPolicy(), sleep=sleep)

        assert await embedder.create('hello') == [0.1, 0.2, 0.3]
        assert inner.calls == 2
        assert sleep.delays == [10.0]

    async def test_create_batch_retries(self):
        inner = FlakyEmbedder([google_resource_exhausted(), google_resource_exhausted()])
        sleep = RecordingSleep()
        embedder = RateLimitRetryEmbedder(inner, RateLimitRetryPolicy(), sleep=sleep)

        assert await embedder.create_batch(['a', 'b']) == [[0.1, 0.2, 0.3]] * 2
        assert inner.calls == 3
        assert sleep.delays == [10.0, 30.0]

    async def test_delegates_attributes_and_is_an_embedder(self):
        inner = FlakyEmbedder([])
        embedder = RateLimitRetryEmbedder(inner, RateLimitRetryPolicy())

        assert embedder.config is inner.config
        assert embedder.inner is inner
        assert isinstance(embedder, EmbedderClient)


class FlakyLLMClient(LLMClient):
    """LLM client whose generate_response rate-limits first and mutates its messages."""

    def __init__(self, failures: int):
        super().__init__(config=None)
        self.failures = failures
        self.seen: list[list[str]] = []

    async def _generate_response(
        self, messages, response_model=None, max_tokens=0, model_size=None
    ):
        raise NotImplementedError

    async def generate_response(self, messages, response_model=None, **kwargs):
        self.seen.append([m.content for m in messages])
        # Provider clients append correction messages while they retry internally.
        messages.append(Message(role='user', content='retry context'))
        if self.failures:
            self.failures -= 1
            raise RateLimitError('Rate limit exceeded. Please try again later.')
        return {'answer': 42, 'kwargs': kwargs}


class TestLLMRateLimitRetry:
    async def test_retries_with_fresh_messages(self):
        sleep = RecordingSleep()
        client = add_llm_rate_limit_retry(
            FlakyLLMClient(failures=2), RateLimitRetryPolicy(), sleep=sleep
        )

        result = await client.generate_response(
            [Message(role='system', content='sys'), Message(role='user', content='hi')],
            group_id='g1',
        )

        assert result == {'answer': 42, 'kwargs': {'group_id': 'g1'}}
        assert sleep.delays == [10.0, 30.0]
        # Every attempt sees the original two messages, not the previous attempt's additions.
        assert client.seen == [['sys', 'hi']] * 3  # type: ignore[attr-defined]

    async def test_keeps_client_type_and_installs_once(self):
        client = FlakyLLMClient(failures=0)
        assert add_llm_rate_limit_retry(client, RateLimitRetryPolicy()) is client
        wrapped = client.generate_response
        add_llm_rate_limit_retry(client, RateLimitRetryPolicy())

        assert isinstance(client, FlakyLLMClient)
        assert client.generate_response is wrapped

    async def test_gives_up_and_raises(self):
        sleep = RecordingSleep()
        client = add_llm_rate_limit_retry(
            FlakyLLMClient(failures=10), RateLimitRetryPolicy(max_retries=2), sleep=sleep
        )

        with pytest.raises(RateLimitError):
            await client.generate_response([Message(role='user', content='hi')])
        assert sleep.delays == [10.0, 30.0]


class TestQueueDeadLetter:
    async def _run_one(self, queue: QueueService, group_id: str, **overrides):
        kwargs = {
            'group_id': group_id,
            'name': 'probe',
            'content': 'Alice met Bob.',
            'source_description': 'unit test',
            'episode_type': EpisodeType.text,
            'entity_types': None,
            'uuid': 'episode-uuid-1',
        }
        kwargs.update(overrides)
        await queue.add_episode(**kwargs)
        await asyncio.wait_for(queue._episode_queues[group_id].join(), timeout=5)

    async def test_failed_episode_is_dead_lettered(self, tmp_path):
        dead_letter = tmp_path / 'dlq' / 'episodes.jsonl'
        client = MagicMock()
        client.add_episode = AsyncMock(side_effect=google_resource_exhausted())
        queue = QueueService(dead_letter_path=dead_letter)
        await queue.initialize(client)

        await self._run_one(queue, 'g1', saga='trip', excluded_entity_types=['Topic'])

        records = [json.loads(line) for line in dead_letter.read_text().splitlines()]
        assert len(records) == 1
        record = records[0]
        assert record['group_id'] == 'g1'
        assert record['uuid'] == 'episode-uuid-1'
        assert record['name'] == 'probe'
        assert record['episode_body'] == 'Alice met Bob.'
        assert record['source'] == 'text'
        assert record['saga'] == 'trip'
        assert record['excluded_entity_types'] == ['Topic']
        assert record['rate_limited'] is True
        assert record['error_type'] == 'ClientError'
        # The episode is attempted once: retries happen at the provider-call level.
        assert client.add_episode.await_count == 1

    async def test_worker_keeps_processing_after_a_failure(self, tmp_path):
        dead_letter = tmp_path / 'episodes.jsonl'
        client = MagicMock()
        client.add_episode = AsyncMock(side_effect=[ValueError('boom'), None])
        queue = QueueService(dead_letter_path=dead_letter)
        await queue.initialize(client)

        await self._run_one(queue, 'g2', uuid='first')
        await self._run_one(queue, 'g2', uuid='second')

        records = [json.loads(line) for line in dead_letter.read_text().splitlines()]
        assert [r['uuid'] for r in records] == ['first']
        assert records[0]['rate_limited'] is False
        assert client.add_episode.await_count == 2

    async def test_successful_episode_is_not_dead_lettered(self, tmp_path):
        dead_letter = tmp_path / 'episodes.jsonl'
        client = MagicMock()
        client.add_episode = AsyncMock(return_value=None)
        queue = QueueService(dead_letter_path=dead_letter)
        await queue.initialize(client)

        await self._run_one(queue, 'g3')

        assert not dead_letter.exists()

    def test_dead_letter_path_from_env(self, monkeypatch, tmp_path):
        monkeypatch.setenv('EPISODE_DEAD_LETTER_PATH', str(tmp_path / 'from-env.jsonl'))
        assert QueueService().dead_letter_path == tmp_path / 'from-env.jsonl'
