import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))

from ingest_wait_helpers import extract_episode_uuid, wait_for_ingest_completion


def test_extract_episode_uuid_from_json_string():
    result = '{"message":"queued","episode_uuid":"episode-1","group_id":"group-1"}'

    assert extract_episode_uuid(result) == 'episode-1'


def test_extract_episode_uuid_from_dict():
    result = {'message': 'queued', 'episode_uuid': 'episode-2', 'group_id': 'group-1'}

    assert extract_episode_uuid(result) == 'episode-2'


def test_extract_episode_uuid_from_mcp_result_object():
    result = SimpleNamespace(
        content=[
            SimpleNamespace(
                text='{"message":"queued","episode_uuid":"episode-3","group_id":"group-1"}'
            )
        ]
    )

    assert extract_episode_uuid(result) == 'episode-3'


def test_extract_episode_uuid_from_mcp_dict_prefers_structured_content():
    result = {
        'result': {
            'content': [{'type': 'text', 'text': '{"message":"queued"}'}],
            'structuredContent': {
                'result': {
                    'message': 'queued',
                    'episode_uuid': 'episode-4',
                    'group_id': 'group-1',
                }
            },
        }
    }

    assert extract_episode_uuid(result) == 'episode-4'


@pytest.mark.asyncio
async def test_wait_for_ingest_completion_succeeds_after_all_complete():
    responses = {
        'episode-1': iter([{'state': 'queued'}, {'state': 'completed'}]),
        'episode-2': iter([{'state': 'processing'}, {'state': 'completed'}]),
    }

    async def _call(tool_name, arguments):
        assert tool_name == 'get_ingest_status'
        return next(responses[arguments['episode_uuid']])

    completed = await wait_for_ingest_completion(
        _call,
        episode_uuids=['episode-1', 'episode-2'],
        group_id='group-1',
        max_wait=1,
        poll_interval=0,
    )

    assert completed is True


@pytest.mark.asyncio
async def test_wait_for_ingest_completion_returns_false_on_failed_episode():
    async def _call(tool_name, arguments):
        return {'state': 'failed', 'last_error': 'boom'}

    completed = await wait_for_ingest_completion(
        _call,
        episode_uuids=['episode-1'],
        group_id='group-1',
        max_wait=1,
        poll_interval=0,
    )

    assert completed is False
