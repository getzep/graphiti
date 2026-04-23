"""Regression tests for response model defaults.

When LLM providers (Anthropic, Gemini) return empty tool input ``{}``,
response models with required list fields crash with ``ValidationError``.
These tests verify that all list-based response models accept empty input
and produce valid instances with empty lists.
"""

from graphiti_core.prompts.dedupe_edges import EdgeDuplicate
from graphiti_core.prompts.dedupe_nodes import NodeDuplicate, NodeResolutions
from graphiti_core.prompts.extract_edges import Edge, ExtractedEdges
from graphiti_core.prompts.extract_nodes import (
    ExtractedEntities,
    ExtractedEntity,
    SummarizedEntities,
    SummarizedEntity,
)


class TestResponseModelsAcceptEmptyInput:
    """Verify that response models handle empty LLM output without crashing.

    This simulates the exact code path in anthropic_client.py (line 403):
        model_instance = response_model(**response)
    where response is {} (empty dict from tool_use input).
    """

    def test_extracted_entities_empty_input(self):
        result = ExtractedEntities(**{})
        assert result.extracted_entities == []

    def test_extracted_edges_empty_input(self):
        result = ExtractedEdges(**{})
        assert result.edges == []

    def test_edge_duplicate_empty_input(self):
        result = EdgeDuplicate(**{})
        assert result.duplicate_facts == []
        assert result.contradicted_facts == []

    def test_node_resolutions_empty_input(self):
        result = NodeResolutions(**{})
        assert result.entity_resolutions == []

    def test_summarized_entities_empty_input(self):
        result = SummarizedEntities(**{})
        assert result.summaries == []


class TestResponseModelsPopulatedInput:
    """Verify that response models still work correctly with populated input."""

    def test_extracted_entities_with_data(self):
        result = ExtractedEntities(
            extracted_entities=[
                ExtractedEntity(name='Alice', entity_type_id=1),
                ExtractedEntity(name='Bob', entity_type_id=2),
            ]
        )
        assert len(result.extracted_entities) == 2
        assert result.extracted_entities[0].name == 'Alice'

    def test_extracted_edges_with_data(self):
        result = ExtractedEdges(
            edges=[
                Edge(
                    source_entity_name='Alice',
                    target_entity_name='Bob',
                    relation_type='KNOWS',
                    fact='Alice knows Bob',
                    valid_at=None,
                    invalid_at=None,
                )
            ]
        )
        assert len(result.edges) == 1
        assert result.edges[0].relation_type == 'KNOWS'

    def test_edge_duplicate_with_data(self):
        result = EdgeDuplicate(duplicate_facts=[1, 2], contradicted_facts=[3])
        assert result.duplicate_facts == [1, 2]
        assert result.contradicted_facts == [3]

    def test_node_resolutions_with_data(self):
        result = NodeResolutions(
            entity_resolutions=[
                NodeDuplicate(id=1, name='Alice', duplicate_candidate_id=-1),
            ]
        )
        assert len(result.entity_resolutions) == 1
        assert result.entity_resolutions[0].duplicate_candidate_id == -1

    def test_summarized_entities_with_data(self):
        result = SummarizedEntities(
            summaries=[
                SummarizedEntity(name='Alice', summary='A person named Alice'),
            ]
        )
        assert len(result.summaries) == 1
        assert result.summaries[0].name == 'Alice'
