from types import SimpleNamespace
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest
from pydantic import BaseModel

from graphiti_core.graphiti_types import GraphitiClients, generate_prompt_response
from graphiti_core.llm_client.client import LLMClient
from graphiti_core.llm_client.config import ModelSize
from graphiti_core.llm_client.llm_runtime import LLMRuntime
from graphiti_core.llm_client.prompt_config import LLMTransport
from graphiti_core.nodes import EpisodeType, EpisodicNode
from graphiti_core.prompts import create_prompt_library, default_chat_prompt_library, prompt_library
from graphiti_core.prompts.extract_nodes import ExtractedEntities
from graphiti_core.prompts.lib import PROMPT_GROUPS
from graphiti_core.prompts.models import ChatPrompt, Message, SystemMessage, UserMessage
from graphiti_core.utils.datetime_utils import utc_now
from graphiti_core.utils.maintenance import node_operations

PROMPT_CONTEXT: dict[str, Any] = {
    'answer': 'Alice knows Bob.',
    'attributes': {'role': 'project lead'},
    'baseline': 'Bob is known by Alice.',
    'candidate': 'Alice knows Bob.',
    'custom_extraction_instructions': 'Use only facts stated in the episode.',
    'edge_invalidation_candidates': '[]',
    'edge_types': [{'name': 'KNOWS', 'description': 'One person knows another.'}],
    'entities': [{'name': 'Alice', 'type': 'Person'}],
    'entity_summaries': [{'name': 'Alice', 'summary': 'Project lead.'}],
    'entity_type_description': {'Person': 'A person mentioned in an episode.'},
    'entity_type_descriptions': {'Person': 'A person mentioned in an episode.'},
    'entity_types': 'Person: A person mentioned in an episode.',
    'episode_content': 'Alice met Bob at the office on January 2.',
    'episodes': ['Alice met Bob on January 2.'],
    'existing_attributes': {'since': '2024'},
    'existing_edges': '[{"fact": "Alice knows Bob."}]',
    'existing_nodes': [{'name': 'Alice', 'summary': 'Project lead.'}],
    'existing_summary': 'Alice is preparing a project launch.',
    'extracted_entities': '[{"name": "Alice", "type": "Person"}]',
    'extracted_node': {'name': 'Alice', 'type': 'Person'},
    'extracted_nodes': [
        {'name': 'Alice', 'type': 'Person'},
        {'name': 'Bob', 'type': 'Person'},
    ],
    'fact': 'Alice knows Bob.',
    'facts': [{'fact': 'Alice knows Bob.'}],
    'message': 'Alice knows Bob.',
    'new_edge': '{"fact": "Alice knows Bob."}',
    'node': 'Alice',
    'node_name': 'Alice',
    'node_summaries': [{'name': 'Alice', 'summary': 'Project lead.'}],
    'node_summary': 'Alice leads the project.',
    'nodes': [{'name': 'Alice', 'summary': 'Project lead.'}],
    'previous_episodes': [{'content': 'Alice met Bob.', 'reference_time': '2025-01-02T00:00:00Z'}],
    'previous_messages': 'User: Who knows Bob?',
    'query': 'Who knows Bob?',
    'reference_time': '2025-01-03T00:00:00Z',
    'response': 'Alice knows Bob.',
    'saga_name': "Alice's project launch",
    'source_description': 'A chat message',
    'summary': 'Alice leads the project.',
}


@pytest.mark.parametrize(
    ('group_name', 'function_name'),
    [
        (group_name, function_name)
        for group_name, function_names in PROMPT_GROUPS.items()
        for function_name in function_names
    ],
)
def test_chat_prompt_defaults_match_legacy_prompt_text(group_name: str, function_name: str) -> None:
    legacy_builder = getattr(getattr(prompt_library, group_name), function_name)
    chat_builder = getattr(getattr(default_chat_prompt_library, group_name), function_name)

    expected = legacy_builder(PROMPT_CONTEXT)
    actual = chat_builder(PROMPT_CONTEXT).as_messages()

    assert [(message.role, message.content.encode('utf-8')) for message in actual] == [
        (message.role, message.content.encode('utf-8')) for message in expected
    ]


@pytest.mark.parametrize(
    'messages',
    [
        [],
        [Message(role='system', content='system')],
        [
            Message(role='user', content='user'),
            Message(role='system', content='system'),
        ],
        [
            Message(role='assistant', content='assistant'),
            Message(role='user', content='user'),
        ],
        [
            Message(role='system', content='system'),
            Message(role='user', content='user'),
            Message(role='user', content='extra'),
        ],
    ],
)
def test_chat_prompt_rejects_invalid_message_shapes(messages: list[Message]) -> None:
    with pytest.raises(ValueError):
        ChatPrompt.from_messages(messages)


def test_chat_prompt_from_messages_copies_content_without_unicode_note() -> None:
    messages = [
        Message(role='system', content='system prompt'),
        Message(role='user', content='user prompt'),
    ]

    prompt = ChatPrompt.from_messages(messages)

    assert prompt.system.content == 'system prompt'
    assert prompt.user.content == 'user prompt'


@pytest.mark.asyncio
@pytest.mark.parametrize('clients', [None, SimpleNamespace()])
async def test_legacy_prompt_response_uses_main_default_kwargs(clients: Any) -> None:
    llm_client = MagicMock(spec=LLMClient)
    llm_client.generate_response = AsyncMock(return_value={'ok': True})
    legacy_prompt = MagicMock(
        return_value=[
            Message(role='system', content='system'),
            Message(role='user', content='user'),
        ]
    )

    result = await generate_prompt_response(
        llm_client,
        'extract_nodes.extract_message',
        legacy_prompt,
        PROMPT_CONTEXT,
        clients=clients,
        response_model=ExtractedEntities,
    )

    assert result == {'ok': True}
    llm_client.generate_response.assert_awaited_once()
    assert llm_client.generate_response.await_args.kwargs == {
        'response_model': ExtractedEntities,
        'prompt_name': 'extract_nodes.extract_message',
    }


@pytest.mark.asyncio
async def test_legacy_prompt_response_forwards_only_non_default_kwargs() -> None:
    llm_client = MagicMock(spec=LLMClient)
    llm_client.generate_response = AsyncMock(return_value={'ok': True})
    messages = [Message(role='system', content='system'), Message(role='user', content='user')]

    await generate_prompt_response(
        llm_client,
        'extract_nodes.extract_message',
        lambda _context: messages,
        PROMPT_CONTEXT,
        response_model=ExtractedEntities,
        max_tokens=128,
        model_size=ModelSize.small,
        group_id='group-a',
        attribute_extraction=True,
    )

    assert llm_client.generate_response.await_args.kwargs == {
        'response_model': ExtractedEntities,
        'prompt_name': 'extract_nodes.extract_message',
        'max_tokens': 128,
        'model_size': ModelSize.small,
        'group_id': 'group-a',
        'attribute_extraction': True,
    }


class LegacySignatureLLMClient(LLMClient):
    def __init__(self) -> None:
        super().__init__(None)
        self.messages: list[Message] | None = None
        self.kwargs: dict[str, Any] = {}

    async def _generate_response(
        self,
        messages: list[Message],
        response_model: type[BaseModel] | None = None,
        max_tokens: int | None = None,
        model_size: ModelSize = ModelSize.medium,
        *,
        model: str | None = None,
    ) -> dict[str, Any]:
        return {}

    async def generate_response(
        self,
        messages: list[Message],
        response_model: type[BaseModel] | None = None,
        max_tokens: int | None = None,
        model_size: ModelSize = ModelSize.medium,
        group_id: str | None = None,
        prompt_name: str | None = None,
        *,
        attribute_extraction: bool = False,
    ) -> dict[str, Any]:
        self.messages = messages
        self.kwargs = {
            'response_model': response_model,
            'max_tokens': max_tokens,
            'model_size': model_size,
            'group_id': group_id,
            'prompt_name': prompt_name,
            'attribute_extraction': attribute_extraction,
        }
        return {'ok': True}


@pytest.mark.asyncio
async def test_legacy_prompt_response_supports_main_llmclient_override_signature() -> None:
    llm_client = LegacySignatureLLMClient()
    messages = [Message(role='system', content='system'), Message(role='user', content='user')]

    await generate_prompt_response(
        llm_client,
        'extract_nodes.extract_message',
        lambda _context: messages,
        PROMPT_CONTEXT,
        response_model=ExtractedEntities,
    )

    assert llm_client.messages == messages
    assert llm_client.kwargs['response_model'] is ExtractedEntities
    assert llm_client.kwargs['prompt_name'] == 'extract_nodes.extract_message'
    assert llm_client.kwargs['max_tokens'] is None
    assert llm_client.kwargs['model_size'] is ModelSize.medium
    assert llm_client.kwargs['group_id'] is None
    assert llm_client.kwargs['attribute_extraction'] is False


@pytest.mark.asyncio
async def test_node_operation_reads_legacy_prompt_builder_at_call_time(monkeypatch) -> None:
    llm_client = MagicMock(spec=LLMClient)
    llm_client.generate_response = AsyncMock(return_value={'extracted_entities': []})
    legacy_prompt = MagicMock(
        return_value=[
            Message(role='system', content='patched system'),
            Message(role='user', content='patched user'),
        ]
    )
    monkeypatch.setattr(
        node_operations.prompt_library.extract_nodes,
        'extract_message',
        legacy_prompt,
    )
    episode = EpisodicNode(
        name='episode',
        source=EpisodeType.message,
        source_description='test',
        content='episode content',
        group_id='group',
        valid_at=utc_now(),
    )
    context = {'episode_content': 'episode content'}

    await node_operations._call_extraction_llm(llm_client, episode, context)

    legacy_prompt.assert_called_once_with(context)
    assert llm_client.generate_response.await_args.args[0][0].content == 'patched system'


def _graphiti_clients(
    llm_client: LLMClient,
    *,
    prompt_library: Any = None,
    llm_runtime: LLMRuntime | None = None,
) -> GraphitiClients:
    return GraphitiClients.model_construct(
        driver=MagicMock(),
        llm_client=llm_client,
        embedder=MagicMock(),
        cross_encoder=MagicMock(),
        tracer=MagicMock(),
        prompt_library=prompt_library,
        llm_runtime=llm_runtime,
    )


@pytest.mark.asyncio
async def test_legacy_prompt_response_uses_configured_prompt_library() -> None:
    llm_client = MagicMock(spec=LLMClient)
    llm_client.generate_response = AsyncMock(return_value={'ok': True})
    library = create_prompt_library(
        {
            'extract_nodes': {
                'extract_message': lambda _context: ChatPrompt(
                    system=SystemMessage(content='custom system'),
                    user=UserMessage(content='custom user'),
                )
            }
        }
    )
    clients = _graphiti_clients(llm_client, prompt_library=library)

    await generate_prompt_response(
        llm_client,
        'extract_nodes.extract_message',
        prompt_library.extract_nodes.extract_message,
        PROMPT_CONTEXT,
        clients=clients,
    )

    llm_client.generate_response.assert_awaited_once()
    message = llm_client.generate_response.await_args.args[0][0]
    assert message.content.startswith('custom system')


@pytest.mark.asyncio
async def test_legacy_prompt_response_uses_configured_runtime() -> None:
    transport_client = MagicMock(spec=LLMClient)
    transport_client.generate_response = AsyncMock(return_value={'ok': True})
    runtime = LLMRuntime(model=LLMTransport(transport_client).model('gpt-4.1'))
    llm_client = MagicMock(spec=LLMClient)
    llm_client.generate_response = AsyncMock(return_value={'unused': True})
    clients = _graphiti_clients(llm_client, llm_runtime=runtime)

    await generate_prompt_response(
        llm_client,
        'extract_nodes.extract_message',
        prompt_library.extract_nodes.extract_message,
        PROMPT_CONTEXT,
        clients=clients,
        model_size=ModelSize.small,
    )

    transport_client.generate_response.assert_awaited_once()
    kwargs = transport_client.generate_response.await_args.kwargs
    assert kwargs['model'] == 'gpt-4.1'
    assert kwargs['model_size'] is ModelSize.small
    llm_client.generate_response.assert_not_awaited()
