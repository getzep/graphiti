from graphiti_core.prompts.dedupe_edges import EdgeDuplicate, resolve_edge


def test_edge_duplicate_schema_collects_reasoning_before_indices():
    fields = list(EdgeDuplicate.model_fields)

    assert fields[:3] == ['reasoning', 'duplicate_facts', 'contradicted_facts']
    assert EdgeDuplicate(
        reasoning='idx 1 contradicts the new fact',
        duplicate_facts=[],
        contradicted_facts=[1],
    ).contradicted_facts == [1]


def test_resolve_edge_prompt_requires_index_reasoning_first():
    prompt = resolve_edge(
        {
            'existing_edges': [{'idx': 0, 'fact': 'A'}],
            'edge_invalidation_candidates': [{'idx': 1, 'fact': 'B'}],
            'new_edge': 'C',
        }
    )[-1].content

    assert 'reasoning' in prompt.lower()
    assert 'idx number' in prompt.lower()
