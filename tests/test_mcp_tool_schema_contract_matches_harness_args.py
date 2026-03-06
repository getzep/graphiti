from scripts.run_retrieval_benchmark import evaluate_mcp_contract


def _tools_response_with_properties(props_by_tool: dict[str, list[str]], required_by_tool=None):
    required_by_tool = required_by_tool or {}
    return {
        'result': {
            'tools': [
                {
                    'name': tool,
                    'inputSchema': {
                        'type': 'object',
                        'properties': {k: {'type': 'string'} for k in props},
                        'required': required_by_tool.get(tool, ['query']),
                    },
                }
                for tool, props in props_by_tool.items()
            ]
        }
    }


def test_contract_check_passes_when_harness_keys_match_schema():
    response = _tools_response_with_properties(
        {
            'search_memory_facts': [
                'query',
                'group_ids',
                'search_mode',
                'max_facts',
                'center_node_uuid',
            ],
            'search_nodes': [
                'query',
                'group_ids',
                'search_mode',
                'max_nodes',
                'entity_types',
            ],
        },
        required_by_tool={
            'search_memory_facts': ['query', 'max_facts'],
            'search_nodes': ['query'],
        },
    )

    contract = evaluate_mcp_contract(tools_list_response=response)
    assert contract['passed'] is True
    assert contract['missing_tools'] == []
    assert contract['unsupported_args'] == {}
    assert contract['missing_required_args'] == {}


def test_contract_check_fails_closed_on_unsupported_harness_arg():
    response = _tools_response_with_properties(
        {
            'search_memory_facts': [
                'query',
                'group_ids',
                'max_facts',
            ],
            'search_nodes': [
                'query',
                'group_ids',
                'max_nodes',
            ],
        },
        required_by_tool={
            'search_memory_facts': ['query', 'max_facts'],
            'search_nodes': ['query'],
        },
    )

    contract = evaluate_mcp_contract(tools_list_response=response)
    assert contract['passed'] is False
    assert contract['unsupported_args']['search_memory_facts'] == ['center_node_uuid', 'search_mode']
    assert contract['unsupported_args']['search_nodes'] == ['entity_types', 'search_mode']
    assert contract['missing_required_args'] == {}
