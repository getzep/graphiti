#!/usr/bin/env python3
"""Reusable MCP benchmark utility for live graphiti-mcp endpoints."""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import time
from typing import Any


def parse_session_id(response_text: str) -> str | None:
    match = re.search(r'mcp-session-id: ([^\r\n]+)', response_text, re.IGNORECASE)
    return match.group(1).strip() if match else None


def parse_sse_data(response_text: str) -> str:
    for line in response_text.splitlines():
        if line.startswith('data: '):
            return line[6:]
    raise ValueError(f'No SSE data line found in response: {response_text[:300]}')


def extract_content_text(payload: dict[str, Any]) -> str:
    try:
        return payload['result']['content'][0]['text']
    except Exception:
        return json.dumps(payload, ensure_ascii=False)


def extract_content_object(payload: dict[str, Any]) -> dict[str, Any]:
    structured_result = payload.get('result', {}).get('structuredContent', {}).get('result')
    if isinstance(structured_result, dict):
        return structured_result

    text = extract_content_text(payload)
    try:
        value = json.loads(text)
    except Exception:
        return {}
    return value if isinstance(value, dict) else {}


def probe_succeeded(nodes_text: str, facts_text: str, expected_terms: list[str]) -> bool:
    if not all(term in nodes_text for term in expected_terms[:1]):
        return False
    return all(term in facts_text or term in nodes_text for term in expected_terms)


def mcp_post(url: str, payload: dict[str, Any], session_id: str | None = None) -> tuple[str | None, dict[str, Any]]:
    cmd = [
        'curl',
        '-i',
        '-sS',
        '--max-time',
        '90',
        '-X',
        'POST',
        url,
        '-H',
        'Content-Type: application/json',
        '-H',
        'Accept: application/json, text/event-stream',
    ]
    if session_id:
        cmd += ['-H', f'mcp-session-id: {session_id}']
    cmd += ['-d', json.dumps(payload, ensure_ascii=False)]

    completed = subprocess.run(cmd, capture_output=True, text=True, check=True)
    response_text = completed.stdout
    next_session = parse_session_id(response_text) or session_id
    return next_session, json.loads(parse_sse_data(response_text))


def build_probe_payload(name: str, group_id: str) -> dict[str, Any]:
    return {
        'name': name,
        'episode_body': (
            'Codex Smoke Tester works at Graphiti Validation Lab and lives in Hangzhou. '
            'Codex Smoke Tester prefers jasmine tea over coffee.'
        ),
        'group_id': group_id,
        'source': 'text',
        'source_description': 'benchmark probe',
    }


def run_benchmark(args) -> dict[str, Any]:
    session_id, _ = mcp_post(
        args.url,
        {
            'jsonrpc': '2.0',
            'id': 1,
            'method': 'initialize',
            'params': {
                'protocolVersion': '2025-03-26',
                'capabilities': {},
                'clientInfo': {'name': 'mcp-benchmark', 'version': '1.0'},
            },
        },
    )

    group_id = f'{args.group_id_prefix}_{int(time.time())}'
    session_id, add_response = mcp_post(
        args.url,
        {
            'jsonrpc': '2.0',
            'id': 2,
            'method': 'tools/call',
            'params': {'name': 'add_memory', 'arguments': build_probe_payload(args.name, group_id)},
        },
        session_id,
    )

    print(f'GROUP_ID={group_id}')
    print(extract_content_text(add_response))
    add_payload = extract_content_object(add_response)
    episode_uuid = add_payload.get('episode_uuid')

    started_at = time.time()
    nodes_text = ''
    facts_text = ''
    success_at_seconds = None
    ingest_state = None

    for attempt in range(1, args.max_attempts + 1):
        time.sleep(args.sleep_seconds)
        elapsed = round(time.time() - started_at, 1)

        if episode_uuid:
            session_id, ingest_response = mcp_post(
                args.url,
                {
                    'jsonrpc': '2.0',
                    'id': 10 + attempt,
                    'method': 'tools/call',
                    'params': {
                        'name': 'get_ingest_status',
                        'arguments': {'episode_uuid': episode_uuid, 'group_id': group_id},
                    },
                },
                session_id,
            )
            ingest_payload = extract_content_object(ingest_response)
            ingest_state = ingest_payload.get('state')
            print(f'T+{elapsed}s INGEST_STATUS[{attempt}] {extract_content_text(ingest_response)[:220]}')

            if ingest_state not in {'completed', 'failed'}:
                continue

            if ingest_state == 'failed':
                break

        session_id, nodes_response = mcp_post(
            args.url,
            {
                'jsonrpc': '2.0',
                'id': 30 + attempt,
                'method': 'tools/call',
                'params': {
                    'name': 'search_nodes',
                    'arguments': {'query': args.query, 'group_ids': [group_id], 'max_nodes': 5},
                },
            },
            session_id,
        )
        nodes_text = extract_content_text(nodes_response)
        print(f'T+{elapsed}s SEARCH_NODES[{attempt}] {nodes_text[:220]}')

        session_id, facts_response = mcp_post(
            args.url,
            {
                'jsonrpc': '2.0',
                'id': 60 + attempt,
                'method': 'tools/call',
                'params': {
                    'name': 'search_memory_facts',
                    'arguments': {
                        'query': args.query,
                        'group_ids': [group_id],
                        'max_facts': 5,
                    },
                },
            },
            session_id,
        )
        facts_text = extract_content_text(facts_response)
        print(f'T+{elapsed}s SEARCH_FACTS[{attempt}] {facts_text[:220]}')

        if probe_succeeded(nodes_text, facts_text, args.expected_term):
            success_at_seconds = elapsed
            break

    if not args.keep_data:
        _, clear_response = mcp_post(
            args.url,
            {
                'jsonrpc': '2.0',
                'id': 99,
                'method': 'tools/call',
                'params': {
                    'name': 'clear_graph',
                    'arguments': {'group_ids': [group_id]},
                },
            },
            session_id,
        )
        print(extract_content_text(clear_response))

    summary = {
        'group_id': group_id,
        'nodes_found': any(term in nodes_text for term in args.expected_term),
        'facts_found': any(term in facts_text for term in args.expected_term),
        'success_at_seconds': success_at_seconds,
        'attempts_used': attempt if 'attempt' in locals() else 0,
        'kept_data': args.keep_data,
        'ingest_state': ingest_state,
    }
    print(json.dumps(summary, ensure_ascii=False))
    return summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description='Benchmark a live graphiti-mcp endpoint')
    parser.add_argument('--url', default='http://127.0.0.1:8000/mcp', help='MCP HTTP endpoint')
    parser.add_argument('--group-id-prefix', default='benchmark', help='Generated group_id prefix')
    parser.add_argument('--name', default='Benchmark memory', help='Episode name')
    parser.add_argument('--query', default='Where does Codex Smoke Tester work and live?', help='Search query')
    parser.add_argument('--sleep-seconds', type=int, default=20, help='Polling interval in seconds')
    parser.add_argument('--max-attempts', type=int, default=18, help='Maximum poll attempts')
    parser.add_argument(
        '--expected-term',
        action='append',
        default=['Codex Smoke Tester', 'Graphiti Validation Lab', 'Hangzhou'],
        help='Expected term for a successful probe; can be passed multiple times',
    )
    parser.add_argument('--keep-data', action='store_true', help='Do not clear the benchmark group')
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    run_benchmark(args)


if __name__ == '__main__':
    main()
