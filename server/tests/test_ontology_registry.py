from graph_service.ontologies import is_known_schema_id, resolve_ontology


def test_is_known_schema_id():
    assert is_known_schema_id('agent_memory_v1') is True
    assert is_known_schema_id('does_not_exist') is False


def test_resolve_ontology_auto_detects_graphiti_episode():
    ontology = resolve_ontology(
        None, '<graphiti_episode kind="memory_directive">...</graphiti_episode>'
    )
    assert ontology is not None
    assert ontology.schema_id == 'agent_memory_v1'


def test_resolve_ontology_uses_default_when_no_schema():
    assert resolve_ontology(None, 'plain message') is None
