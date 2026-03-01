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
import urllib.error
import urllib.request
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

MCP_URL_DEFAULT = 'http://localhost:8000/mcp'
SEARCH_MODES = ['hybrid', 'semantic', 'keyword']
MODE_TIEBREAK_ORDER = {'hybrid': 0, 'semantic': 1, 'keyword': 2}

# Hard cap on bytes read from a single MCP HTTP response.  Prevents the benchmark
# from consuming unbounded memory when the server returns an unexpectedly large
# payload (e.g. returning thousands of facts without a server-side limit).
_MAX_RESPONSE_BYTES = 4 * 1024 * 1024  # 4 MiB

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
                # Bounded read: guard against unexpectedly large server responses.
                # Read cap+1 bytes so we can distinguish "exactly cap" (ok) from "over cap"
                # (overflow). A response of exactly _MAX_RESPONSE_BYTES is valid.
                raw = resp.read(_MAX_RESPONSE_BYTES + 1)
                if len(raw) > _MAX_RESPONSE_BYTES:
                    return {'error': {'code': -1, 'message': 'response exceeded max size'}}
                body = raw.decode('utf-8', errors='replace')
                sid = resp.headers.get('mcp-session-id')
                if sid and not self.session_id:
                    self.session_id = sid
                if 'text/event-stream' in ct:
                    lines = [
                        line[len('data:'):].strip()
                        for line in body.splitlines()
                        if line.startswith('data:')
                    ]
                    return json.loads(lines[-1]) if lines else {}
                return json.loads(body) if body.strip() else {}
        except urllib.error.HTTPError as e:
            raw_err = e.read(_MAX_RESPONSE_BYTES)
            body = raw_err.decode('utf-8', errors='replace')
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
    # Fail hard if MCP returned an error — do not silently score garbage.
    if isinstance(facts_resp, dict) and 'error' in facts_resp:
        raise RuntimeError(
            f'MCP returned error for query {query!r} (search_memory_facts): '
            f'{str(facts_resp["error"])[:200]}'
        )
    if not isinstance(facts_resp, (list, dict)):
        raise RuntimeError(
            f'MCP returned unexpected type {type(facts_resp).__name__} '
            f'for query {query!r} (search_memory_facts)'
        )
    # Check for tool-level errors in content (ErrorResponse nested in result.content[].text)
    if isinstance(facts_resp, dict) and 'content' in facts_resp.get('result', facts_resp):
        _content_check = facts_resp.get('result', facts_resp)
        for _item in _content_check.get('content', []):
            if isinstance(_item, dict):
                _text = _item.get('text', '')
                if isinstance(_text, str) and '"error"' in _text:
                    try:
                        _parsed = json.loads(_text)
                        if isinstance(_parsed, dict) and 'error' in _parsed:
                            raise RuntimeError(
                                f'MCP tool returned ErrorResponse for query {query!r} '
                                f'(search_memory_facts): {str(_parsed["error"])[:200]}'
                            )
                    except json.JSONDecodeError:
                        pass

    node_args: dict[str, Any] = {'query': query, 'max_nodes': top_k}
    if lane_alias:
        node_args['lane_alias'] = lane_alias
    if search_mode != 'hybrid':
        node_args['search_mode'] = search_mode
    nodes_resp = client.call_tool('search_nodes', node_args)
    # Fail hard if MCP returned an error — do not silently score garbage.
    if isinstance(nodes_resp, dict) and 'error' in nodes_resp:
        raise RuntimeError(
            f'MCP returned error for query {query!r} (search_nodes): '
            f'{str(nodes_resp["error"])[:200]}'
        )
    if not isinstance(nodes_resp, (list, dict)):
        raise RuntimeError(
            f'MCP returned unexpected type {type(nodes_resp).__name__} '
            f'for query {query!r} (search_nodes)'
        )
    # Check for tool-level errors in content (ErrorResponse nested in result.content[].text)
    if isinstance(nodes_resp, dict) and 'content' in nodes_resp.get('result', nodes_resp):
        _content_check = nodes_resp.get('result', nodes_resp)
        for _item in _content_check.get('content', []):
            if isinstance(_item, dict):
                _text = _item.get('text', '')
                if isinstance(_text, str) and '"error"' in _text:
                    try:
                        _parsed = json.loads(_text)
                        if isinstance(_parsed, dict) and 'error' in _parsed:
                            raise RuntimeError(
                                f'MCP tool returned ErrorResponse for query {query!r} '
                                f'(search_nodes): {str(_parsed["error"])[:200]}'
                            )
                    except json.JSONDecodeError:
                        pass

    return {
        'facts_response': facts_resp,
        'nodes_response': nodes_resp,
        'facts_text': extract_results_text(facts_resp),
        'nodes_text': extract_results_text(nodes_resp),
    }


_QMD_TIMEOUT_SECONDS = 120  # QMD loads large ML models per invocation; allow generous budget


def run_qmd_query(qmd_command: str, query: str) -> dict:
    """Run a QMD query via subprocess.

    Raises RuntimeError when QMD fails or returns invalid JSON output.
    The timeout is set generously (``_QMD_TIMEOUT_SECONDS``) to accommodate
    QMD's per-invocation model-load overhead (embedding + reranker models).
    """
    cmd_parts = shlex.split(qmd_command) + ['--', query]
    try:
        result = subprocess.run(
            cmd_parts,
            shell=False,
            capture_output=True,
            text=True,
            timeout=_QMD_TIMEOUT_SECONDS,
            check=True,
        )
    except subprocess.CalledProcessError as exc:
        stderr = (exc.stderr or '').strip()
        raise RuntimeError(
            f'QMD query failed with exit code {exc.returncode}: {stderr or "<no stderr>"}'
        ) from exc
    except subprocess.TimeoutExpired as exc:
        raise RuntimeError(
            f'QMD query timed out after {_QMD_TIMEOUT_SECONDS} seconds'
        ) from exc
    except FileNotFoundError as exc:
        raise RuntimeError(f'QMD command not found: {cmd_parts[0]}') from exc

    try:
        parsed = json.loads(result.stdout)
    except json.JSONDecodeError as exc:
        raise RuntimeError(
            f'QMD query returned invalid JSON: {exc.msg} (pos {exc.pos})'
        ) from exc

    return {'parsed': parsed, 'text': result.stdout}


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


def check_recall_gate(
    results: dict,
    threshold: float,
    *,
    baseline_path: str | None = None,
) -> dict:
    """Evaluate the recall non-regression gate.

    Compares the current run's mean combined recall against:
    1. A fixed ``threshold`` (absolute floor, e.g. 0.70).
    2. A ``baseline_path`` JSON from a previous run (regression delta check).

    Returns a dict with keys:
      passed (bool), score (float), threshold (float),
      baseline_score (float|None), delta (float|None), details (str)

    Does NOT raise — callers decide whether to sys.exit(1) based on ``passed``.
    """
    agg = results.get('bicameral_aggregate', {})
    try:
        score: float = float(agg.get('mean_combined_recall_at_k', 0.0))
    except (TypeError, ValueError):
        score = 0.0

    baseline_score: float | None = None
    delta: float | None = None

    if baseline_path:
        bp = Path(baseline_path)
        if bp.exists():
            try:
                baseline_data = json.loads(bp.read_text(encoding='utf-8'))
                baseline_agg = baseline_data.get('bicameral_aggregate', {})
                raw_baseline = baseline_agg.get('mean_combined_recall_at_k')
                if raw_baseline is not None:
                    try:
                        baseline_score = float(raw_baseline)
                    except (TypeError, ValueError) as coerce_exc:
                        print(
                            f'[recall-gate] WARNING: malformed baseline score '
                            f'{raw_baseline!r} — ignoring ({coerce_exc})'
                        )
                if baseline_score is not None:
                    delta = score - baseline_score
            except Exception as exc:
                print(f'[recall-gate] WARNING: could not read baseline {baseline_path}: {exc}')

    passed = score >= threshold
    if delta is not None and delta < 0:
        # Regression: score dropped vs baseline — fail regardless of threshold
        passed = False

    details_parts = [f'score={score:.4f}', f'threshold={threshold:.4f}']
    if baseline_score is not None:
        details_parts.append(f'baseline={baseline_score:.4f}')
        details_parts.append(f'delta={delta:+.4f}')
    details_parts.append('PASS' if passed else 'FAIL')

    return {
        'passed': passed,
        'score': score,
        'threshold': threshold,
        'baseline_score': baseline_score,
        'delta': delta,
        'details': ' | '.join(details_parts),
    }


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
    # ── Recall non-regression gate (Phase C / Slice 4) ──────────────────────
    ap.add_argument(
        '--recall-gate',
        type=float,
        default=None,
        metavar='THRESHOLD',
        help=(
            'Fail (exit 1) if mean combined recall@k falls below THRESHOLD '
            '(0.0–1.0). Omit to skip the gate check.'
        ),
    )
    ap.add_argument(
        '--recall-baseline',
        default=None,
        metavar='PATH',
        help=(
            'Path to a previous benchmark JSON.  If provided, the gate also '
            'fails if the current score regresses vs the baseline.'
        ),
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

    # Recall non-regression gate
    if args.recall_gate is not None:
        gate = check_recall_gate(
            results,
            threshold=args.recall_gate,
            baseline_path=args.recall_baseline,
        )
        print(f'\n[recall-gate] {gate["details"]}')
        if not gate['passed']:
            print('[recall-gate] FAILED — exiting with code 1')
            import sys
            sys.exit(1)
        print('[recall-gate] PASSED')


if __name__ == '__main__':
    main()
