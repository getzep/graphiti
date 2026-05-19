from graph_service.config import Settings
from graph_service.zep_graphiti import build_graphiti_clients


def test_build_graphiti_clients_applies_openai_base_url_to_clients():
    settings = Settings(
        openai_api_key='test_key',
        openai_base_url='http://example.test/v1',
        model_name='gpt-4o-mini',
        embedding_model_name='text-embedding-3-small',
        neo4j_uri='bolt://neo4j:7687',
        neo4j_user='neo4j',
        neo4j_password='password',
    )

    llm_client, embedder, cross_encoder = build_graphiti_clients(settings)

    assert str(llm_client.client.base_url).rstrip('/') == 'http://example.test/v1'
    assert str(embedder.client.base_url).rstrip('/') == 'http://example.test/v1'
    assert str(cross_encoder.client.base_url).rstrip('/') == 'http://example.test/v1'
    assert embedder.config.embedding_model == 'text-embedding-3-small'
