"""OpenAI-compatible client adapters for providers with partial OpenAI support."""

import json
import logging
import typing
from typing import Any

import openai
from openai.types.chat import ChatCompletionMessageParam
from pydantic import BaseModel, ValidationError

from graphiti_core.llm_client.config import DEFAULT_MAX_TOKENS, LLMConfig, ModelSize
from graphiti_core.llm_client.errors import RateLimitError
from graphiti_core.llm_client.openai_generic_client import DEFAULT_MODEL, OpenAIGenericClient
from graphiti_core.prompts.models import Message
from graphiti_core.llm_client.client import get_extraction_language_instruction


logger = logging.getLogger(__name__)


class OpenAICompatibleJSONClient(OpenAIGenericClient):
    """Use json_object mode for OpenAI-compatible providers that do not support json_schema."""

    @staticmethod
    def _extract_json_payload(content: str) -> str:
        """Extract the first JSON object or array from provider content."""
        stripped = content.strip()
        if '\\n' in stripped and '```' in stripped:
            stripped = stripped.replace('\\n', '\n')
        if stripped.startswith('```'):
            lines = [line for line in stripped.splitlines() if not line.strip().startswith('```')]
            stripped = '\n'.join(lines).strip()

        if stripped.startswith('{') or stripped.startswith('['):
            return stripped

        obj_start = stripped.find('{')
        arr_start = stripped.find('[')
        starts = [idx for idx in (obj_start, arr_start) if idx != -1]
        if not starts:
            return stripped

        return stripped[min(starts) :].strip()

    @staticmethod
    def _normalize_single_field_answer_wrapper(
        parsed: Any, response_model: type[BaseModel]
    ) -> dict[str, Any]:
        """Map provider-specific `answer` wrappers to single-field response schemas."""
        model_fields = response_model.model_fields
        if len(model_fields) != 1:
            return parsed

        field_name = next(iter(model_fields))
        if isinstance(parsed, list):
            return {field_name: parsed}

        if isinstance(parsed, dict) and 'answer' in parsed:
            normalized = dict(parsed)
            normalized[field_name] = parsed['answer']
            return normalized

        if isinstance(parsed, dict) and field_name not in parsed:
            list_like_keys = [key for key, value in parsed.items() if isinstance(value, list)]
            if len(list_like_keys) == 1:
                normalized = dict(parsed)
                normalized[field_name] = parsed[list_like_keys[0]]
                return normalized

        return parsed

    async def _generate_response(
        self,
        messages: list[Message],
        response_model: type[BaseModel] | None = None,
        max_tokens: int = DEFAULT_MAX_TOKENS,
        model_size: ModelSize = ModelSize.medium,
    ) -> dict[str, Any]:
        openai_messages: list[ChatCompletionMessageParam] = []
        for message in messages:
            message.content = self._clean_input(message.content)
            if message.role == 'user':
                openai_messages.append({'role': 'user', 'content': message.content})
            elif message.role == 'system':
                openai_messages.append({'role': 'system', 'content': message.content})

        try:
            response = await self.client.chat.completions.create(
                model=self.model or DEFAULT_MODEL,
                messages=openai_messages,
                temperature=self.temperature,
                max_tokens=self.max_tokens,
                response_format={'type': 'json_object'},
            )

            choices = getattr(response, 'choices', None)
            if not choices:
                raise ValueError(f'Provider returned no choices: {response}')

            content = choices[0].message.content
            if content is None:
                raise ValueError(f'Provider returned null message content: {response}')

            parsed = json.loads(self._extract_json_payload(content))
            if response_model is not None:
                try:
                    return response_model.model_validate(parsed).model_dump()
                except ValidationError:
                    normalized = self._normalize_single_field_answer_wrapper(
                        parsed, response_model
                    )
                    return response_model.model_validate(normalized).model_dump()
            return parsed
        except openai.RateLimitError as exc:
            raise RateLimitError from exc
        except json.JSONDecodeError as exc:
            self.logger.error(
                'Error in generating LLM response: %s\n%s',
                exc,
                self._get_failed_generation_log(messages, content if 'content' in locals() else None),
            )
            raise
        except Exception as exc:
            self.logger.error(f'Error in generating LLM response: {exc}')
            raise

    def __init__(
        self,
        config: LLMConfig | None = None,
        cache: bool = False,
        client: Any = None,
        max_tokens: int = 16384,
    ):
        super().__init__(config=config, cache=cache, client=client, max_tokens=max_tokens)
        self.logger = logger

    async def generate_response(
        self,
        messages: list[Message],
        response_model: type[BaseModel] | None = None,
        max_tokens: int | None = None,
        model_size: ModelSize = ModelSize.medium,
        group_id: str | None = None,
        prompt_name: str | None = None,
    ) -> dict[str, typing.Any]:
        if max_tokens is None:
            max_tokens = self.max_tokens

        if response_model is not None:
            serialized_model = json.dumps(response_model.model_json_schema())
            messages[
                -1
            ].content += (
                f'\n\nRespond with a JSON object in the following format:\n\n{serialized_model}'
            )

        messages[0].content += get_extraction_language_instruction(group_id)

        for message in messages:
            message.content = self._clean_input(message.content)

        return await super().generate_response(
            messages,
            response_model=response_model,
            max_tokens=max_tokens,
            model_size=model_size,
            group_id=None,
            prompt_name=prompt_name,
        )
