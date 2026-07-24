import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from graphiti_core.edges import EntityEdge
from graphiti_core.nodes import EntityNode

from utils.formatting import format_fact_result, to_node_result


def _node():
    return EntityNode(
        uuid='n1',
        name='FanWeb',
        group_id='g',
        labels=['Entity'],
        created_at=datetime.now(timezone.utc),
        summary='s',
    )


def _edge():
    return EntityEdge(
        uuid='e1',
        name='USES',
        fact='A uses B',
        group_id='g',
        source_node_uuid='n1',
        target_node_uuid='n2',
        created_at=datetime.now(timezone.utc),
    )


def test_node_result_includes_score():
    r = to_node_result(_node(), score=0.42)
    assert r['score'] == 0.42


def test_node_result_score_defaults_none():
    r = to_node_result(_node())
    assert r['score'] is None


def test_fact_result_includes_score():
    r = format_fact_result(_edge(), score=0.99)
    assert r['score'] == 0.99
