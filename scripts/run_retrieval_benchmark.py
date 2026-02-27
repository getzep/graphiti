#!/usr/bin/env python3
"""Retrieval benchmark harness: Bicameral lane-targeted search vs QMD.

Runs a fixed set of known-answer query pairs and produces structured
comparison output with per-query and aggregate recall@k metrics.

Usage:
  python3 scripts/run_retrieval_benchmark.py \
    --fixture tests/fixtures/retrieval_benchmark_queries.json \
    --top-k 10 --output state/retrieval_benchmark_results.json

  # With QMD comparison:
  python3 scripts/run_retrieval_benchmark.py \
    --fixture tests/fixtures/retrieval_benchmark_queries.json \
    --compare-qmd --qmd-command "qmd query --json" \
    --top-k 10 --output state/retrieval_benchmark_comparison.json
"""

from __future__ import annotations

import argparse
import json
import shlex
import subprocess
import sys
import time
import urllib.error
import urllib.request
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

MCP_URL_DEFAULT = 'http://localhost:8000/mcp'
SEARCH_MODES = ['hybrid', 'semantic', 'keyword']
MODE_TIEBREAK_ORDER = {'hybrid': 0, 'semantic': 1, 'keyword': 2}

# Minimum fixture quotas by lane category
FIXTURE_QUOTAS = {
    'sessions_main': 6,
    'observational_memory': 6,
    'curated': 6,
    'chatgpt': 4,
    'cross_lane': 8,
}


class BenchmarkMCPClient:
    """Minimal MCP client for benchmark queries."""

    def __init__(self, url: str = MCP_URL_DEFAULT):
        self.url = url
        self.session_id: str | None = None
        self.initialized = False

    def _http_post(self, payload: dict, extra_headers: dict | None = None) -> dict:
        data = json.dumps(payload).encode('utf-8')
        headers = {
            'Accept': 'application/json, text/event-stream',
            'Content-Type': 'application/json',
        }
        if extra_headers:
            headers.update(extra_headers)

        req = urllib.request.Request(self.url, data=data, headers=headers, method='POST')
        try:
            with urllib.request.urlopen(req, timeout=60) as resp:
                ct = resp.headers.get('content-type', '')
                body = resp.read().decode('utf-8', errors='replace')
                sid = resp.headers.get('mcp-session-id')
                if sid and not self.session_id:
                    self.session_id = sid
                if 'text/event-stream' in ct:
                    lines = [
                        l[len('data:'):].strip()
                        for l in body.splitlines()
                        if l.startswith('data:')
                    ]
                    return json.loads(lines[-1]) if lines else {}
                return json.loads(body) if body.strip() else {}
        except urllib.error.HTTPError as e:
            body = e.read().decode('utf-8', errors='replace')
            sid = e.headers.get('mcp-session-id') if e.headers else None
            if sid and not self.session_id:
                self.session_id = sid
            if e.code == 400 and 'Missing session ID' in body and self.session_id:
                return self._http_post(
                    payload, {'mcp-session-id': self.session_id}
                )
            return {'error': {'code': e.code, 'message': body}}

    def _post(self, payload: dict) -> dict:
        extra = {}
        if self.session_id:
            extra['mcp-session-id'] = self.session_id
        return self._http_post(payload, extra if extra else None)

    def initialize(self):
        if self.initialized:
            return
        self._post({
            'jsonrpc': '2.0', 'id': 1, 'method': 'initialize',
            'params': {
                'protocolVersion': '2024-11-05',
                'capabilities': {},
                'clientInfo': {'name': 'retrieval-benchmark', 'version': '1'},
            },
        })
        self._http_post(
            {'jsonrpc': '2.0', 'method': 'notifications/initialized', 'params': {}},
            {'mcp-session-id': self.session_id} if self.session_id else None,
        )
        self.initialized = True

    def call_tool(self, name: str, arguments: dict) -> dict:
        self.initialize()
        return self._post({
            'jsonrpc': '2.0', 'id': 1,
            'method': 'tools/call',
            'params': {'name': name, 'arguments': arguments},
        })


def compute_recall(results_text: str, expected: list[str]) -> float:
    """Compute recall of expected items found in results text."""
    if not expected:
        return 1.0
    text_lower = results_text.lower()
    found = sum(1 for e in expected if e.lower() in text_lower)
    return found / len(expected)


def extract_results_text(mcp_response: dict) -> str:
    """Extract searchable text from MCP tool response."""
    result = mcp_response.get('result', {})
    content = result.get('content', [])
    parts = []
    for item in content:
        if isinstance(item, dict):
            text = item.get('text', '')
            if text:
                parts.append(text)
    if not parts:
        # Fallback: serialize the whole response
        parts.append(json.dumps(mcp_response))
    return '\n'.join(parts)


def run_bicameral_query(
    client: BenchmarkMCPClient,
    query: str,
    lane_alias: list[str] | None,
    search_mode: str,
    top_k: int,
) -> dict:
    """Run a single Bicameral search query and return results."""
    args: dict[str, Any] = {'query': query, 'max_facts': top_k}
    if lane_alias:
        args['lane_alias'] = lane_alias
    if search_mode != 'hybrid':
        args['search_mode'] = search_mode

    facts_resp = client.call_tool('search_memory_facts', args)

    node_args: dict[str, Any] = {'query': query, 'max_nodes': top_k}
    if lane_alias:
        node_args['lane_alias'] = lane_alias
    if search_mode != 'hybrid':
        node_args['search_mode'] = search_mode
    nodes_resp = client.call_tool('search_nodes', node_args)

    return {
        'facts_response': facts_resp,
        'nodes_response': nodes_resp,
        'facts_text': extract_results_text(facts_resp),
        'nodes_text': extract_results_text(nodes_resp),
    }


def run_qmd_query(qmd_command: str, query: str) -> dict:
    """Run a QMD query via subprocess."""
    cmd_parts = shlex.split(qmd_command) + [query]
    try:
        result = subprocess.run(
            cmd_parts,
            shell=False,
            capture_output=True,
            text=True,
            timeout=30,
        )
        if result.returncode != 0:
            return {'error': result.stderr, 'text': ''}
        try:
            parsed = json.loads(result.stdout)
            return {'parsed': parsed, 'text': result.stdout}
        except json.JSONDecodeError:
            return {'text': result.stdout}
    except subprocess.TimeoutExpired:
        return {'error': 'timeout', 'text': ''}
    except FileNotFoundError:
        return {'error': f'command not found: {cmd_parts[0]}', 'text': ''}


def validate_fixture(queries: list[dict]) -> list[str]:
    """Validate fixture meets coverage quotas."""
    errors = []
    if len(queries) < 30:
        errors.append(f'fixture must have >= 30 queries, got {len(queries)}')

    # Count by category
    counts: dict[str, int] = {
        'sessions_main': 0,
        'observational_memory': 0,
        'curated': 0,
        'chatgpt': 0,
        'cross_lane': 0,
    }
    for q in queries:
        aliases = q.get('lane_alias', [])
        if len(aliases) > 1:
            counts['cross_lane'] += 1
        elif aliases:
            alias = aliases[0]
            if alias in counts:
                counts[alias] += 1

    for category, quota in FIXTURE_QUOTAS.items():
        if counts[category] < quota:
            errors.append(
                f'{category} queries: need >= {quota}, got {counts[category]}'
            )

    return errors


def run_benchmark(args: argparse.Namespace) -> dict:
    """Run the full benchmark and return results."""
    fixture_path = Path(args.fixture)
    if not fixture_path.exists():
        raise SystemExit(f'Fixture file not found: {fixture_path}')

    queries = json.loads(fixture_path.read_text(encoding='utf-8'))
    fixture_errors = validate_fixture(queries)
    if fixture_errors:
        print('Fixture validation errors:')
        for e in fixture_errors:
            print(f'  - {e}')
        raise SystemExit(1)

    top_k = args.top_k
    do_qmd = args.compare_qmd

    # Initialize MCP client
    client = BenchmarkMCPClient(args.mcp_url)

    query_results = []
    bicameral_scores = []
    qmd_scores = []

    for qi, q in enumerate(queries, 1):
        qid = q['id']
        query_text = q['query']
        expected_facts = q.get('expected_facts', [])
        expected_entities = q.get('expected_entities', [])
        lane_alias = q.get('lane_alias')

        print(f'[{qi}/{len(queries)}] {qid}: {query_text[:60]}...')

        mode_scores = {}
        for mode in SEARCH_MODES:
            try:
                result = run_bicameral_query(
                    client, query_text, lane_alias, mode, top_k
                )
                combined_text = result['facts_text'] + '\n' + result['nodes_text']
                fact_recall = compute_recall(combined_text, expected_facts)
                entity_recall = compute_recall(combined_text, expected_entities)
                combined = (fact_recall + entity_recall) / 2.0
                mode_scores[mode] = {
                    'fact_recall_at_k': round(fact_recall, 4),
                    'entity_recall_at_k': round(entity_recall, 4),
                    'combined_recall_at_k': round(combined, 4),
                }
            except Exception as exc:
                print(f'  ERROR [{mode}]: {exc}')
                raise

        # Select best mode (tiebreak: hybrid > semantic > keyword)
        best_mode = max(
            mode_scores.keys(),
            key=lambda m: (
                mode_scores[m]['combined_recall_at_k'],
                -MODE_TIEBREAK_ORDER[m],
            ),
        )
        best_score = mode_scores[best_mode]['combined_recall_at_k']
        bicameral_scores.append(best_score)

        qr: dict[str, Any] = {
            'id': qid,
            'query': query_text,
            'bicameral': {
                'mode_scores': mode_scores,
                'best_mode': best_mode,
                'best_score': best_score,
            },
        }

        # QMD comparison
        if do_qmd:
            qmd_result = run_qmd_query(args.qmd_command, query_text)
            qmd_text = qmd_result.get('text', '')
            qmd_fact_recall = compute_recall(qmd_text, expected_facts)
            qmd_entity_recall = compute_recall(qmd_text, expected_entities)
            qmd_combined = (qmd_fact_recall + qmd_entity_recall) / 2.0
            qmd_scores.append(qmd_combined)
            qr['qmd'] = {
                'score': round(qmd_combined, 4),
                'fact_recall_at_k': round(qmd_fact_recall, 4),
                'entity_recall_at_k': round(qmd_entity_recall, 4),
            }
            qr['delta_vs_qmd'] = round(best_score - qmd_combined, 4)
        else:
            qr['delta_vs_qmd'] = None

        query_results.append(qr)

    # Aggregate
    bicameral_agg = {
        'mean_combined_recall_at_k': round(
            sum(bicameral_scores) / len(bicameral_scores), 4
        ) if bicameral_scores else 0.0,
        'queries_evaluated': len(bicameral_scores),
    }

    output: dict[str, Any] = {
        'fixture_path': str(fixture_path),
        'top_k': top_k,
        'timestamp': datetime.now(timezone.utc).isoformat().replace('+00:00', 'Z'),
        'queries_total': len(queries),
        'bicameral_aggregate': bicameral_agg,
        'query_results': query_results,
    }

    if do_qmd and qmd_scores:
        output['qmd_aggregate'] = {
            'mean_combined_recall_at_k': round(
                sum(qmd_scores) / len(qmd_scores), 4
            ),
            'queries_evaluated': len(qmd_scores),
        }

    return output


def main():
    ap = argparse.ArgumentParser(
        description='Retrieval benchmark: Bicameral vs QMD'
    )
    ap.add_argument('--fixture', required=True, help='Path to fixture JSON')
    ap.add_argument('--top-k', type=int, default=10, help='Top-k for recall')
    ap.add_argument('--output', required=True, help='Output JSON path')
    ap.add_argument(
        '--mcp-url', default=MCP_URL_DEFAULT, help='MCP server URL'
    )
    ap.add_argument(
        '--compare-qmd', action='store_true', help='Enable QMD comparison'
    )
    ap.add_argument(
        '--qmd-command',
        default='qmd query --json',
        help='QMD command template (default: qmd query --json)',
    )
    args = ap.parse_args()

    results = run_benchmark(args)

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(results, indent=2, ensure_ascii=False) + '\n',
        encoding='utf-8',
    )

    print(f'\nResults written to {output_path}')
    agg = results['bicameral_aggregate']
    print(f'Bicameral mean recall@{args.top_k}: {agg["mean_combined_recall_at_k"]}')
    if 'qmd_aggregate' in results:
        qagg = results['qmd_aggregate']
        print(f'QMD mean recall@{args.top_k}: {qagg["mean_combined_recall_at_k"]}')


if __name__ == '__main__':
    main()
