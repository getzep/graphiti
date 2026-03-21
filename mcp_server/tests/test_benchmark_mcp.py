"""Unit tests for the reusable MCP benchmark script."""

import json
from argparse import Namespace

import benchmark_mcp
from benchmark_mcp import extract_content_text, parse_session_id, parse_sse_data, probe_succeeded


def test_parse_session_id_extracts_header_value():
    response = (
        'HTTP/1.1 200 OK\r\n'
        'content-type: text/event-stream\r\n'
        'mcp-session-id: abc123\r\n'
        '\r\n'
        'event: message\r\n'
        'data: {"jsonrpc":"2.0"}\r\n'
    )

    assert parse_session_id(response) == 'abc123'


def test_parse_session_id_returns_none_when_missing():
    response = 'HTTP/1.1 200 OK\r\ncontent-type: text/event-stream\r\n\r\n'

    assert parse_session_id(response) is None


def test_parse_sse_data_extracts_first_data_line():
    response = (
        'HTTP/1.1 200 OK\r\n\r\n'
        'event: message\r\n'
        'data: {"jsonrpc":"2.0","id":1}\r\n'
        'data: {"ignored":true}\r\n'
    )

    assert parse_sse_data(response) == '{"jsonrpc":"2.0","id":1}'


def test_parse_sse_data_raises_when_missing():
    response = 'HTTP/1.1 200 OK\r\n\r\nevent: message\r\n'

    try:
        parse_sse_data(response)
    except ValueError as exc:
        assert 'No SSE data line found' in str(exc)
    else:
        raise AssertionError('Expected parse_sse_data() to raise ValueError')


def test_extract_content_text_from_tool_response():
    payload = {
        'result': {
            'content': [{'type': 'text', 'text': '{"message":"ok"}'}],
        }
    }

    assert extract_content_text(payload) == '{"message":"ok"}'


def test_extract_content_object_prefers_structured_content_result():
    payload = {
        'result': {
            'content': [{'type': 'text', 'text': '{"message":"queued"}'}],
            'structuredContent': {
                'result': {
                    'message': 'queued',
                    'episode_uuid': 'episode-structured',
                    'group_id': 'group-1',
                    'queue_position': 1,
                }
            },
        }
    }

    assert benchmark_mcp.extract_content_object(payload) == {
        'message': 'queued',
        'episode_uuid': 'episode-structured',
        'group_id': 'group-1',
        'queue_position': 1,
    }


def test_probe_succeeded_requires_node_and_fact_hits():
    assert probe_succeeded(
        nodes_text='Codex Smoke Tester works at Graphiti Validation Lab',
        facts_text='Codex Smoke Tester lives in Hangzhou and prefers jasmine tea.',
        expected_terms=['Codex Smoke Tester', 'Graphiti Validation Lab', 'Hangzhou'],
    )

    assert not probe_succeeded(
        nodes_text='No relevant nodes found',
        facts_text='Codex Smoke Tester lives in Hangzhou.',
        expected_terms=['Codex Smoke Tester', 'Graphiti Validation Lab', 'Hangzhou'],
    )


def test_run_benchmark_prefers_ingest_status_before_search(monkeypatch):
    calls = []
    responses = iter(
        [
            {'result': {'content': [{'text': '{"message":"initialized"}'}]}},
            {
                'result': {
                    'content': [
                        {
                            'text': json.dumps(
                                {
                                    'message': 'queued',
                                    'episode_uuid': 'episode-1',
                                    'group_id': 'benchmark_123',
                                    'queue_position': 1,
                                }
                            )
                        }
                    ]
                }
            },
            {'result': {'content': [{'text': '{"state":"queued","episode_uuid":"episode-1"}'}]}},
            {
                'result': {
                    'content': [{'text': '{"state":"completed","episode_uuid":"episode-1"}'}]
                }
            },
            {
                'result': {
                    'content': [
                        {
                            'text': 'Codex Smoke Tester works at Graphiti Validation Lab'
                        }
                    ]
                }
            },
            {
                'result': {
                    'content': [{'text': 'Codex Smoke Tester lives in Hangzhou'}]
                }
            },
        ]
    )

    def _fake_mcp_post(url, payload, session_id=None):
        calls.append(payload['params']['name'] if payload['method'] == 'tools/call' else payload['method'])
        return 'session-1', next(responses)

    monkeypatch.setattr(benchmark_mcp, 'mcp_post', _fake_mcp_post)
    monkeypatch.setattr(benchmark_mcp.time, 'sleep', lambda _: None)
    monkeypatch.setattr(
        benchmark_mcp.time,
        'time',
        iter([1234567890.0, 1234567890.0, 1234567890.0, 1234567890.0]).__next__,
    )

    summary = benchmark_mcp.run_benchmark(
        Namespace(
            url='http://localhost:8000/mcp',
            group_id_prefix='benchmark',
            name='Benchmark memory',
            query='Where does Codex Smoke Tester work and live?',
            sleep_seconds=0,
            max_attempts=2,
            expected_term=['Codex Smoke Tester', 'Graphiti Validation Lab', 'Hangzhou'],
            keep_data=True,
        )
    )

    assert summary['success_at_seconds'] == 0.0
    assert calls[:6] == [
        'initialize',
        'add_memory',
        'get_ingest_status',
        'get_ingest_status',
        'search_nodes',
        'search_memory_facts',
    ]
