#!/usr/bin/env python3
"""Utility-style Bicameral vs QMD evaluation harness.

Generates three artifacts:
- JSON summary with per-query winner + rationale
- Markdown report
- Markdown worksheet for manual calibration
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.run_retrieval_benchmark import (  # noqa: E402
    BenchmarkMCPClient,
    _query_scope_group_ids,
    compute_recall,
    run_bicameral_query,
    run_qmd_query,
)


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace('+00:00', 'Z')


def _load_fixture(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        raise FileNotFoundError(f'fixture not found: {path}')
    raw = json.loads(path.read_text(encoding='utf-8'))
    if not isinstance(raw, list):
        raise ValueError('fixture must be a list of query objects')
    return raw


def _score_output(text: str, expected_facts: list[str], expected_entities: list[str]) -> dict[str, float]:
    fact_recall = compute_recall(text, expected_facts)
    entity_recall = compute_recall(text, expected_entities)
    utility_score = (fact_recall + entity_recall) / 2.0
    return {
        'fact_recall_at_k': round(fact_recall, 4),
        'entity_recall_at_k': round(entity_recall, 4),
        'utility_score': round(utility_score, 4),
    }


def _winner_and_rationale(bicameral_score: float, qmd_score: float) -> tuple[str, str]:
    delta = round(bicameral_score - qmd_score, 4)
    if delta > 0:
        return ('bicameral', f'Bicameral utility higher by {delta:.4f}')
    if delta < 0:
        return ('qmd', f'QMD utility higher by {abs(delta):.4f}')
    return ('tie', 'Utility scores tied')


def evaluate_utility_vs_qmd(
    *,
    fixture: list[dict[str, Any]],
    top_k: int,
    mcp_url: str,
    qmd_command: str,
    max_queries: int | None = None,
) -> dict[str, Any]:
    client = BenchmarkMCPClient(mcp_url)
    per_query: list[dict[str, Any]] = []

    bicameral_scores: list[float] = []
    qmd_scores: list[float] = []
    winner_counts = {'bicameral': 0, 'qmd': 0, 'tie': 0}

    selected_fixture = fixture[:max_queries] if max_queries and max_queries > 0 else fixture

    for index, query_item in enumerate(selected_fixture, start=1):
        qid = str(query_item.get('id') or f'q{index:03d}')
        
        print(f'[{index}/{len(selected_fixture)}] {qid}: evaluating utility vs qmd')
        query = str(query_item.get('query') or '').strip()
        if not query:
            raise ValueError(f'fixture row {qid} missing query')

        expected_facts = query_item.get('expected_facts', []) or []
        expected_entities = query_item.get('expected_entities', []) or []
        target_group_ids = _query_scope_group_ids(query_item)

        bicameral = run_bicameral_query(
            client,
            query=query,
            group_ids=target_group_ids,
            search_mode='hybrid',
            top_k=top_k,
        )
        bicameral_text = (bicameral['facts_text'] + '\n' + bicameral['nodes_text']).strip()
        bicameral_eval = _score_output(bicameral_text, expected_facts, expected_entities)

        qmd = run_qmd_query(qmd_command, query)
        qmd_text = str(qmd.get('text') or '')
        qmd_eval = _score_output(qmd_text, expected_facts, expected_entities)

        winner, rationale = _winner_and_rationale(
            bicameral_eval['utility_score'],
            qmd_eval['utility_score'],
        )
        winner_counts[winner] += 1
        bicameral_scores.append(bicameral_eval['utility_score'])
        qmd_scores.append(qmd_eval['utility_score'])

        per_query.append(
            {
                'id': qid,
                'query': query,
                'target_group_ids': target_group_ids,
                'bicameral': bicameral_eval,
                'qmd': qmd_eval,
                'winner': winner,
                'score_delta_bicameral_minus_qmd': round(
                    bicameral_eval['utility_score'] - qmd_eval['utility_score'],
                    4,
                ),
                'scoring_rationale': rationale,
            }
        )

    aggregate = {
        'queries_evaluated': len(per_query),
        'bicameral_mean_utility': round(sum(bicameral_scores) / len(bicameral_scores), 4)
        if bicameral_scores
        else 0.0,
        'qmd_mean_utility': round(sum(qmd_scores) / len(qmd_scores), 4) if qmd_scores else 0.0,
        'winner_counts': winner_counts,
        'net_advantage': round(
            (sum(bicameral_scores) / len(bicameral_scores) if bicameral_scores else 0.0)
            - (sum(qmd_scores) / len(qmd_scores) if qmd_scores else 0.0),
            4,
        ),
    }

    return {
        'timestamp': _utc_now(),
        'top_k': top_k,
        'fixture_total': len(fixture),
        'queries_selected': len(selected_fixture),
        'per_query': per_query,
        'aggregate': aggregate,
    }


def _render_report(data: dict[str, Any], fixture_path: str) -> str:
    agg = data['aggregate']
    lines = [
        '# Utility Eval vs QMD',
        '',
        f"- generated_at: {data['timestamp']}",
        f"- fixture: `{fixture_path}`",
        f"- fixture_total: {data.get('fixture_total')}",
        f"- queries_selected: {data.get('queries_selected')}",
        f"- queries_evaluated: {agg['queries_evaluated']}",
        f"- bicameral_mean_utility: {agg['bicameral_mean_utility']}",
        f"- qmd_mean_utility: {agg['qmd_mean_utility']}",
        f"- net_advantage (bicameral-qmd): {agg['net_advantage']}",
        f"- winner_counts: {agg['winner_counts']}",
        '',
        '## Per-query winners',
        '',
    ]

    for row in data['per_query']:
        lines.extend(
            [
                f"- **{row['id']}** winner={row['winner']} delta={row['score_delta_bicameral_minus_qmd']}",
                f"  - rationale: {row['scoring_rationale']}",
            ]
        )

    return '\n'.join(lines) + '\n'


def _render_worksheet(data: dict[str, Any]) -> str:
    lines = [
        '# Utility Eval Worksheet',
        '',
        'Use this worksheet for spot-checking automated utility winners.',
        '',
        '| Query ID | Winner | Bicameral Utility | QMD Utility | Human override (optional) | Notes |',
        '|---|---:|---:|---:|---|---|',
    ]

    for row in data['per_query']:
        lines.append(
            f"| {row['id']} | {row['winner']} | {row['bicameral']['utility_score']:.4f} "
            f"| {row['qmd']['utility_score']:.4f} |  | {row['scoring_rationale']} |"
        )

    return '\n'.join(lines) + '\n'


def main() -> int:
    ap = argparse.ArgumentParser(description='Utility eval vs QMD from live retrieval outputs')
    ap.add_argument('--fixture', required=True, help='Benchmark fixture JSON')
    ap.add_argument('--output-json', required=True, help='Output JSON path')
    ap.add_argument('--output-md', required=True, help='Output markdown report path')
    ap.add_argument('--worksheet-md', required=True, help='Output worksheet markdown path')
    ap.add_argument('--mcp-url', default='http://localhost:8000/mcp', help='MCP server URL')
    ap.add_argument('--qmd-command', default='qmd query --json', help='QMD command template')
    ap.add_argument('--top-k', type=int, default=10, help='top-k for Bicameral retrieval')
    ap.add_argument('--max-queries', type=int, default=None, help='Optional cap for sampled evaluation runs')
    args = ap.parse_args()

    fixture_path = Path(args.fixture)
    fixture = _load_fixture(fixture_path)
    data = evaluate_utility_vs_qmd(
        fixture=fixture,
        top_k=args.top_k,
        mcp_url=args.mcp_url,
        qmd_command=args.qmd_command,
        max_queries=args.max_queries,
    )

    out_json = Path(args.output_json)
    out_md = Path(args.output_md)
    out_ws = Path(args.worksheet_md)
    for path in (out_json, out_md, out_ws):
        path.parent.mkdir(parents=True, exist_ok=True)

    out_json.write_text(json.dumps(data, indent=2) + '\n', encoding='utf-8')
    out_md.write_text(_render_report(data, str(fixture_path)), encoding='utf-8')
    out_ws.write_text(_render_worksheet(data), encoding='utf-8')

    print(f'utility eval JSON: {out_json}')
    print(f'utility eval report: {out_md}')
    print(f'utility eval worksheet: {out_ws}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
