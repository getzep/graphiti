"""Scenario harness for edge duplicate-candidate resolution.

Runs synthetic end-to-end scenarios through resolve_extracted_edges with a
scripted search stack and a deterministic text-overlap "LLM judge" (which,
like the real dedupe prompt, sees only fact text -- never endpoints), and
records what the LLM was shown and what got merged.

Two uses:

1. As pytest tests (CI): each scenario asserts the current contract --
   the duplicate-candidate search is pre-filtered to the extracted edge's
   endpoints (never graph-wide), only the invalidation search is
   graph-wide, and a fact is never merged into an edge between unrelated
   nodes.

2. As a cross-version A/B reporter: run the file directly, pointing
   PYTHONPATH at any graphiti checkout, and diff the printed reports::

       git worktree add /tmp/zep-baseline <ref>
       PYTHONPATH=/tmp/zep-baseline/graphiti uv run --no-sync \
           python tests/utils/maintenance/test_edge_dedup_scenarios.py
       uv run python tests/utils/maintenance/test_edge_dedup_scenarios.py

   The report prints which source tree was loaded. Against pre-2026-06
   code this reproduces the historical failure modes: the duplicate search
   degenerating to a graph-wide scan on empty candidate sets, fact merges
   across unrelated node pairs, and the invalidation-candidate list being
   emptied by the overlap dedup ("cannibalization").

The fake search emulates endpoint pre-filtering for the current design and
intentionally keeps the *old* production filter semantics for edge_uuids
(a falsy list drops the filter -> graph-wide retrieval) so baseline runs
against pre-fix refs reproduce what production actually did.
"""

import ast
import asyncio
import hashlib
import math
import re
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

from graphiti_core.edges import EntityEdge
from graphiti_core.nodes import EntityNode, EpisodicNode
from graphiti_core.search.search_config import SearchResults
from graphiti_core.utils.maintenance import edge_operations as edge_ops

NOW = datetime(2026, 6, 10, tzinfo=timezone.utc)


def _text_vec(text: str) -> list[float]:
    """Deterministic bag-of-words unit vector, a stand-in embedder."""
    v = [0.0] * 64
    for tok in re.findall(r'\w+', text.lower()):
        v[int(hashlib.md5(tok.encode(), usedforsecurity=False).hexdigest(), 16) % 64] += 1.0
    norm = math.sqrt(sum(x * x for x in v)) or 1.0
    return [x / norm for x in v]


def _dot(a: list[float], b: list[float]) -> float:
    return sum(x * y for x, y in zip(a, b, strict=True))


def _jaccard(a: str, b: str) -> float:
    ta = set(re.findall(r'\w+', a.lower()))
    tb = set(re.findall(r'\w+', b.lower()))
    return len(ta & tb) / (len(ta | tb) or 1)


def _make_edge(uuid: str, src: str, tgt: str, fact: str, age_days: int = 1) -> EntityEdge:
    return EntityEdge(
        uuid=uuid,
        source_node_uuid=src,
        target_node_uuid=tgt,
        name='RELATES_TO',
        group_id='g1',
        fact=fact,
        episodes=['old_episode'],
        created_at=NOW - timedelta(days=age_days),
        valid_at=NOW - timedelta(days=age_days),
        invalid_at=None,
    )


SCENARIOS = {
    'cross_endpoint_identical_text': {
        # The hazard: an identical fact between a *different* node pair must
        # not absorb the new fact (the prompt-level judge cannot see
        # endpoints, so offering it as a duplicate candidate merges it).
        'graph': [_make_edge('E1', 'dan_jones', 'yoga', 'Dan likes yoga')],
        'extracted': ('dan_smith', 'yoga', 'Dan likes yoga'),
        'nodes': ['dan_jones', 'dan_smith', 'yoga'],
    },
    'same_pair_duplicate': {
        # Parity: an exact same-pair duplicate resolves via the fast path
        # with no LLM call.
        'graph': [_make_edge('E2', 'user', 'yoga', 'User likes yoga')],
        'extracted': ('user', 'yoga', '  user LIKES yoga '),
        'nodes': ['user', 'yoga'],
    },
    'reverse_direction_duplicate': {
        # The undirected candidate fetch must catch a duplicate stored in
        # the opposite orientation.
        'graph': [_make_edge('E3', 'bob', 'alice', 'Alice manages Bob')],
        'extracted': ('alice', 'bob', 'Alice manages Bob'),
        'nodes': ['alice', 'bob'],
    },
    'contradiction_on_new_pair': {
        # A contradicting fact on a new node pair must reach the LLM as a
        # properly labeled invalidation candidate (historically this list
        # was emptied by the overlap dedup whenever the duplicate search
        # degenerated to graph-wide).
        'graph': [_make_edge('E4', 'alice', 'acme', 'Alice works at Acme')],
        'extracted': ('alice', 'betacorp', 'Alice works at BetaCorp'),
        'nodes': ['alice', 'acme', 'betacorp'],
    },
}


async def run_scenario(scenario: dict) -> dict:
    graph_edges: list[EntityEdge] = scenario['graph']
    src, tgt, fact = scenario['extracted']
    stats: dict = {
        'searches': [],
        'existing_facts_shown': None,
        'cross_endpoint_shown': None,
        'invalidation_shown': None,
        'llm_called': False,
    }

    async def fake_search(clients, query, group_ids=None, config=None, search_filter=None, **kw):
        source_ids = getattr(search_filter, 'edge_source_node_uuids', None)
        target_ids = getattr(search_filter, 'edge_target_node_uuids', None)
        uuids = getattr(search_filter, 'edge_uuids', None)
        if source_ids or target_ids:
            # Endpoint pre-filter (current design): ranking runs over the
            # endpoint-scoped subset only -- never the whole graph.
            pool = [
                e
                for e in graph_edges
                if (not source_ids or e.source_node_uuid in set(source_ids))
                and (not target_ids or e.target_node_uuid in set(target_ids))
            ]
            stats['searches'].append('pair-filtered')
        elif uuids:
            pool = [e for e in graph_edges if e.uuid in set(uuids)]
            stats['searches'].append('filtered')
        else:
            # Old production semantics on purpose: falsy uuid list (None OR
            # empty) -> unfiltered graph-wide retrieval, so baseline runs
            # against pre-fix refs stay honest. See module docstring.
            pool = list(graph_edges)
            stats['searches'].append('graph-wide')
        query_vec = _text_vec(query)
        ranked = sorted(pool, key=lambda e: -_dot(_text_vec(e.fact), query_vec))
        return SearchResults(edges=ranked[:10])

    async def fake_between(driver, s, t, **kw):
        # Only exercised by baseline runs against refs that still fetch
        # between-nodes candidates; the current design searches instead.
        return [e for e in graph_edges if (e.source_node_uuid, e.target_node_uuid) == (s, t)]

    def parse_list(message: str, tag: str) -> list[dict]:
        m = re.search(rf'<{tag}>\s*(\[.*?\])\s*</{tag}>', message, re.S)
        return ast.literal_eval(m.group(1)) if m else []

    async def fake_llm(messages, response_model=None, prompt_name='', **kw):
        if 'resolve_edge' not in prompt_name:
            return {}
        stats['llm_called'] = True
        body = messages[-1].content
        existing = parse_list(body, 'EXISTING FACTS')
        invalidation = parse_list(body, 'FACT INVALIDATION CANDIDATES')
        new_fact_match = re.search(r'<NEW FACT>\s*(.*?)\s*</NEW FACT>', body, re.S)
        new_fact = new_fact_match.group(1) if new_fact_match else fact
        stats['existing_facts_shown'] = [e['fact'] for e in existing]
        by_fact = {e.fact: e for e in graph_edges}
        stats['cross_endpoint_shown'] = sum(
            1
            for e in existing
            if e['fact'] in by_fact
            and (by_fact[e['fact']].source_node_uuid, by_fact[e['fact']].target_node_uuid)
            != (src, tgt)
        )
        stats['invalidation_shown'] = [e['fact'] for e in invalidation]
        duplicates = [e['idx'] for e in existing if _jaccard(e['fact'], new_fact) >= 0.8]
        contradicted = [
            e['idx'] for e in existing + invalidation if 0.4 <= _jaccard(e['fact'], new_fact) < 0.8
        ]
        return {'duplicate_facts': duplicates, 'contradicted_facts': contradicted}

    llm_client = MagicMock()
    llm_client.generate_response = AsyncMock(side_effect=fake_llm)
    embedder = MagicMock()
    embedder.create_batch = AsyncMock(side_effect=lambda texts: [_text_vec(t) for t in texts])

    extracted = EntityEdge(
        source_node_uuid=src,
        target_node_uuid=tgt,
        name='RELATES_TO',
        group_id='g1',
        fact=fact,
        episodes=[],
        created_at=NOW,
        valid_at=NOW,  # set so _extract_edge_timestamps skips its LLM call
        invalid_at=None,
    )
    episode = EpisodicNode(
        uuid='ep1',
        name='ep',
        group_id='g1',
        source='message',
        source_description='',
        content='',
        valid_at=NOW,
    )
    nodes = [
        EntityNode(uuid=u, name=u, group_id='g1', labels=['Entity']) for u in scenario['nodes']
    ]
    clients = SimpleNamespace(
        driver=MagicMock(),
        llm_client=llm_client,
        embedder=embedder,
        cross_encoder=MagicMock(),
    )

    with (
        patch.object(edge_ops, 'search', side_effect=fake_search),
        patch.object(EntityEdge, 'get_between_nodes', AsyncMock(side_effect=fake_between)),
    ):
        resolved, invalidated, _ = await edge_ops.resolve_extracted_edges(
            clients, [extracted], episode, nodes, {}, {}
        )

    result = resolved[0]
    return {
        'searches': stats['searches'],
        'existing_facts': stats['existing_facts_shown'],
        'cross_endpoint_candidates': stats['cross_endpoint_shown'],
        'invalidation_candidates': stats['invalidation_shown'],
        'llm_called': stats['llm_called'],
        'resolved_uuid': result.uuid,
        'extracted_uuid': extracted.uuid,
        'merged_cross_endpoint': (result.source_node_uuid, result.target_node_uuid) != (src, tgt),
        'invalidated': len(invalidated),
    }


async def test_cross_endpoint_identical_text_is_not_merged():
    report = await run_scenario(SCENARIOS['cross_endpoint_identical_text'])
    assert report['searches'] == ['pair-filtered', 'graph-wide']
    assert report['existing_facts'] == [], 'no duplicate candidates from other node pairs'
    assert report['merged_cross_endpoint'] is False
    assert report['resolved_uuid'] == report['extracted_uuid']
    # The similar fact still reaches the LLM, properly labeled.
    assert report['invalidation_candidates'] == ['Dan likes yoga']


async def test_same_pair_duplicate_fast_path_one_search():
    report = await run_scenario(SCENARIOS['same_pair_duplicate'])
    assert report['resolved_uuid'] == 'E2'
    assert report['llm_called'] is False, 'exact match resolves without an LLM call'
    assert report['searches'] == ['pair-filtered', 'graph-wide']


async def test_reverse_direction_duplicate_is_caught():
    report = await run_scenario(SCENARIOS['reverse_direction_duplicate'])
    assert report['resolved_uuid'] == 'E3'
    assert report['searches'] == ['pair-filtered', 'graph-wide']


async def test_contradiction_on_new_pair_reaches_invalidation_list():
    report = await run_scenario(SCENARIOS['contradiction_on_new_pair'])
    assert report['invalidation_candidates'] == ['Alice works at Acme'], (
        'invalidation candidates must not be cannibalized by duplicate selection'
    )
    assert report['existing_facts'] == []
    assert report['invalidated'] == 1
    assert report['merged_cross_endpoint'] is False


async def _report() -> None:
    print(f'graphiti_core loaded from: {edge_ops.__file__}\n')
    for name, scenario in SCENARIOS.items():
        result = await run_scenario(scenario)
        print(f'== {name}')
        for key, value in result.items():
            print(f'   {key}: {value}')
        print()


if __name__ == '__main__':
    asyncio.run(_report())
