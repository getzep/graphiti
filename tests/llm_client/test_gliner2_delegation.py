import importlib
from typing import Any

import pytest
from pydantic import BaseModel

pytest.importorskip('gliner2', reason='GLiNER2 is an optional dependency')

client_module = importlib.import_module('graphiti_core.llm_client.client')
config_module = importlib.import_module('graphiti_core.llm_client.config')
gliner2_module = importlib.import_module('graphiti_core.llm_client.gliner2_client')
prompts_module = importlib.import_module('graphiti_core.prompts.models')

LLMClient = client_module.LLMClient
LLMConfig = config_module.LLMConfig
Message = prompts_module.Message
ModelSize = config_module.ModelSize


class SummaryResponse(BaseModel):
    summary: str


class LegacyLLMClient(LLMClient):
    def __init__(self) -> None:
        super().__init__(LLMConfig())
        self.call: dict[str, Any] | None = None

    async def _generate_response(
        self,
        messages: list[Message],
        response_model: type[BaseModel] | None = None,
        max_tokens: int | None = None,
        model_size: ModelSize = ModelSize.medium,
    ) -> dict[str, Any]:
        raise NotImplementedError

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
        self.call = {
            'messages': messages,
            'response_model': response_model,
            'max_tokens': max_tokens,
            'model_size': model_size,
            'group_id': group_id,
            'prompt_name': prompt_name,
            'attribute_extraction': attribute_extraction,
        }
        return {'result': 'legacy delegate'}


class ModelAwareLLMClient(LegacyLLMClient):
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
        model: str | None = None,
    ) -> dict[str, Any]:
        response = await super().generate_response(
            messages,
            response_model=response_model,
            max_tokens=max_tokens,
            model_size=model_size,
            group_id=group_id,
            prompt_name=prompt_name,
            attribute_extraction=attribute_extraction,
        )
        if self.call is not None:
            self.call['model'] = model
        return response


def make_gliner2_client(monkeypatch: pytest.MonkeyPatch, llm_client: LLMClient) -> Any:
    class StubGLiNER2:
        @staticmethod
        def from_pretrained(model_id: str) -> object:
            return object()

    monkeypatch.setattr(gliner2_module, 'GLiNER2', StubGLiNER2)
    return gliner2_module.GLiNER2Client(llm_client=llm_client)


@pytest.mark.asyncio
async def test_non_extraction_delegation_omits_unset_model_for_legacy_client(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    delegate = LegacyLLMClient()
    client = make_gliner2_client(monkeypatch, delegate)
    messages = [Message(role='system', content='system'), Message(role='user', content='user')]

    result = await client.generate_response(messages, response_model=SummaryResponse)

    assert result == {'result': 'legacy delegate'}
    assert delegate.call is not None
    assert delegate.call['response_model'] is SummaryResponse


@pytest.mark.asyncio
async def test_non_extraction_delegation_forwards_set_model(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    delegate = ModelAwareLLMClient()
    client = make_gliner2_client(monkeypatch, delegate)
    messages = [Message(role='system', content='system'), Message(role='user', content='user')]

    await client.generate_response(messages, response_model=SummaryResponse, model='x')

    assert delegate.call is not None
    assert delegate.call['model'] == 'x'
