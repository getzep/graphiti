import asyncio

from tests.helpers_mcp_import import load_search_service

SearchService = load_search_service().SearchService


class _FakeNeo4jService:
    async def search_om_nodes(self, *_args, **_kwargs):
        return [
            {
                'uuid': 'om-node-1',
                'content': 'Observed recurring morning routine',
                'created_at': '2026-03-05T00:00:00Z',
                'group_id': 's1_observational_memory',
                'status': 'active',
            }
        ]

    async def search_om_facts(self, *_args, **_kwargs):
        return [
            {
                'uuid': 'om-fact-1',
                'relation_type': 'observed_pattern',
                'source_node_id': 'om-node-1',
                'target_node_id': 'om-node-2',
                'source_content': 'Morning workout block before 10:30',
                'target_content': 'Daily routine adherence',
                'created_at': '2026-03-05T00:00:00Z',
                'group_id': 's1_observational_memory',
            }
        ]


class _FakeClient:
    driver = object()


class _FakeGraphitiService:
    class config:  # noqa: D401 - minimal shim for provider lookup
        class database:
            provider = 'neo4j'

    async def get_client(self):
        return _FakeClient()


def test_om_only_scope_returns_om_evidence():
    async def _run():
        service = SearchService(neo4j_service=_FakeNeo4jService())
        graphiti = _FakeGraphitiService()

        facts = await service.search_observational_facts(
            graphiti_service=graphiti,
            query='What recurring observations exist?',
            group_ids=['s1_observational_memory'],
            max_facts=5,
            center_node_uuid=None,
        )

        assert len(facts) == 1
        fact = facts[0]
        assert fact['group_id'] == 's1_observational_memory'
        assert fact['attributes']['source'] == 'om_primitive'

    asyncio.run(_run())
