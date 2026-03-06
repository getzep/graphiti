from tests.helpers_mcp_import import load_graphiti_mcp_server

_fuse_node_like_results = load_graphiti_mcp_server()._fuse_node_like_results


def _item(*, uuid: str, summary: str, source: str) -> dict:
    return {
        'uuid': uuid,
        'name': f'fact-{uuid}',
        'summary': summary,
        'fact': f'fact-{uuid}',
        'attributes': {'source': source},
        'group_id': 's1_observational_memory',
    }


def test_fusion_prefers_graphiti_payload_on_uuid_overlap():
    primary = [
        _item(uuid='g1', summary='primary graphiti row', source='graphiti'),
    ]
    supplemental = [
        _item(uuid='g1', summary='OM duplicate row', source='om'),
        _item(uuid='g2', summary='OM only row', source='om'),
    ]

    fused = _fuse_node_like_results(
        primary=primary,
        supplemental=supplemental,
        max_items=3,
    )

    assert len(fused) == 2
    ids = [row['uuid'] for row in fused]
    assert ids == ['g1', 'g2']
    assert fused[0]['summary'] == 'primary graphiti row'
    assert fused[0]['attributes']['source'] == 'graphiti'


def test_fusion_requires_non_empty_inputs_and_respects_cap():
    primary = [_item(uuid='p1', summary='alpha', source='graphiti')]
    supplemental = [_item(uuid='o1', summary='bravo', source='om')]

    fused = _fuse_node_like_results(
        primary=primary,
        supplemental=supplemental,
        max_items=1,
    )
    assert len(fused) == 1
    assert fused[0]['uuid'] == 'p1'

    fused_many = _fuse_node_like_results(
        primary=primary,
        supplemental=supplemental,
        max_items=2,
    )
    assert len(fused_many) == 2
