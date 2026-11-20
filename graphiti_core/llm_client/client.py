"""
Copyright 2024, Zep Software, Inc.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

import contextvars
import hashlib
import json
import logging
import typing
from abc import ABC, abstractmethod

import httpx
from pydantic import BaseModel
from tenacity import retry, retry_if_exception, stop_after_attempt, wait_random_exponential

from ..prompts.models import Message
from ..prompts.versions import UNKNOWN_PROMPT_VERSION, PromptTemplate
from ..tracer import NoOpTracer, Tracer
from .cache import LLMCache
from .config import DEFAULT_MAX_TOKENS, LLMConfig, ModelSize
from .errors import EmptyResponseError, RateLimitError
from .token_tracker import TokenUsageTracker

DEFAULT_CACHE_DIR = './llm_cache'

# current_prompt_name carries the prompt_name of the in-flight generate_response
# call to downstream _generate_response implementations (e.g. gateway
# clients, which add it to the request for inference tracing) without
# threading the argument through every provider signature. It is an asyncio
# task-local, so concurrent extractions don't clobber each other.
current_prompt_name: contextvars.ContextVar[str | None] = contextvars.ContextVar(
    'graphiti_current_prompt_name', default=None
)
current_prompt_version: contextvars.ContextVar[str | None] = contextvars.ContextVar(
    'graphiti_current_prompt_version', default=None
)

ATTRIBUTE_EXTRACTION_PREAMBLE = (
    '\n\n<<graphiti.attr_extraction.preamble.v1>>\n'
    'ATTRIBUTE EXTRACTION: Field descriptions in the response schema describe '
    'what a real value LOOKS LIKE — they are NEVER themselves valid values and '
    'must NEVER be copied into any field. If you have no value for a field, set '
    'it to null; never explain the absence in the field itself.'
)
RESPONSE_FORMAT_TEMPLATE = (
    '\n\nRespond with a JSON object in the following format:\n\n{serialized_model}'
)
LANGUAGE_INSTRUCTION_TEMPLATE = (
    '\n\nAny extracted information should be returned in the same language as it was written in. '
    'Only output non-English text when the user has written full sentences or phrases in that '
    'non-English language. Otherwise, output English.'
)


def get_extraction_language_instruction(group_id: str | None = None) -> str:
    """Returns instruction for language extraction behavior.

    Override this function to customize language extraction:
    - Return empty string to disable multilingual instructions
    - Return custom instructions for specific language requirements
    - Use group_id to provide different instructions per group/partition

    Args:
        group_id: Optional partition identifier for the graph

    Returns:
        str: Language instruction to append to system messages
    """
    return LANGUAGE_INSTRUCTION_TEMPLATE


logger = logging.getLogger(__name__)


def is_server_or_retry_error(exception):
    # EmptyResponseError is treated as transient: an empty body is most often a flaky
    # provider/endpoint hiccup (common on the OpenAI-compatible/local servers the generic
    # client targets), which a retry can recover from. A persistent empty response still
    # fails after the bounded retries with a clear error.
    if isinstance(exception, RateLimitError | EmptyResponseError | json.decoder.JSONDecodeError):
        return True

    return (
        isinstance(exception, httpx.HTTPStatusError) and 500 <= exception.response.status_code < 600
    )


class LLMClient(ABC):
    # When True, node dedupe sends one request per unresolved node with only that
    # node's candidates, in place of one request with the union of all candidates.
    dedupe_nodes_per_entity: bool = False

    def __init__(self, config: LLMConfig | None, cache: bool = False):
        if config is None:
            config = LLMConfig()

        self.config = config
        self.model = config.model
        self.small_model = config.small_model
        self.temperature = config.temperature
        self.max_tokens = config.max_tokens
        self.cache_enabled = cache
        self.cache_dir = None
        self.tracer: Tracer = NoOpTracer()
        self.token_tracker: TokenUsageTracker = TokenUsageTracker()

        # Only create the cache directory if caching is enabled
        if self.cache_enabled:
            self.cache_dir = LLMCache(DEFAULT_CACHE_DIR)

    def set_tracer(self, tracer: Tracer) -> None:
        """Set the tracer for this LLM client."""
        self.tracer = tracer

    def _clean_input(self, input: str) -> str:
        """Clean input string of invalid unicode and control characters.

        Args:
            input: Raw input string to be cleaned

        Returns:
            Cleaned string safe for LLM processing
        """
        # Clean any invalid Unicode
        cleaned = input.encode('utf-8', errors='ignore').decode('utf-8')

        # Remove zero-width characters and other invisible unicode
        zero_width = '\u200b\u200c\u200d\ufeff\u2060'
        for char in zero_width:
            cleaned = cleaned.replace(char, '')

        # Remove control characters except newlines, returns, and tabs
        cleaned = ''.join(char for char in cleaned if ord(char) >= 32 or char in '\n\r\t')

        return cleaned

    @retry(
        stop=stop_after_attempt(4),
        wait=wait_random_exponential(multiplier=10, min=5, max=120),
        retry=retry_if_exception(is_server_or_retry_error),
        after=lambda retry_state: (
            logger.warning(
                f'Retrying {retry_state.fn.__name__ if retry_state.fn else "function"} after {retry_state.attempt_number} attempts...'
            )
            if retry_state.attempt_number > 1
            else None
        ),
        reraise=True,
    )
    async def _generate_response_with_retry(
        self,
        messages: list[Message],
        response_model: type[BaseModel] | None = None,
        max_tokens: int = DEFAULT_MAX_TOKENS,
        model_size: ModelSize = ModelSize.medium,
        *,
        model: str | None = None,
    ) -> dict[str, typing.Any]:
        try:
            # Only splat overrides when set so third-party subclasses with the
            # pre-existing four-param signature still work on the legacy path.
            overrides: dict[str, typing.Any] = {}
            if model is not None:
                overrides['model'] = model
            return await self._generate_response(
                messages, response_model, max_tokens, model_size, **overrides
            )
        except (httpx.HTTPStatusError, RateLimitError) as e:
            raise e

    @abstractmethod
    async def _generate_response(
        self,
        messages: list[Message],
        response_model: type[BaseModel] | None = None,
        max_tokens: int = DEFAULT_MAX_TOKENS,
        model_size: ModelSize = ModelSize.medium,
        *,
        model: str | None = None,
    ) -> dict[str, typing.Any]:
        pass

    def _get_cache_key(self, messages: list[Message], model: str | None = None) -> str:
        # Create a unique cache key based on the messages and model
        message_str = json.dumps([m.model_dump() for m in messages], sort_keys=True)
        key_str = f'{model or self.model}:{message_str}'
        return hashlib.md5(key_str.encode(), usedforsecurity=False).hexdigest()

    def _apply_attribute_extraction_preamble(
        self, messages: list[Message], attribute_extraction: bool
    ) -> None:
        """Append a strict-framing instruction to the system message for attribute
        extraction calls.

        Customer-supplied entity-type schemas use ``Field(description=...)`` to describe
        the FORMAT of a value (e.g. "Phone numbers, comma-separated"). Models — across
        OpenAI, Anthropic, and Gemini — have been observed copying that description text
        verbatim into the value when no real value exists, or dumping reasoning instead
        of returning null. This preamble names the failure mode explicitly so the
        instruction reaches every provider regardless of how structured output is wired
        (schema injection in the prompt body, ``response_format``, native tool use, etc.).

        Mutates ``messages`` in place. Idempotent so concrete-provider overrides can
        safely call it without coordinating with the base class.
        """
        if not attribute_extraction or not messages:
            return
        # Unique sentinel so the idempotency check can't collide with prompt content
        # that happens to mention "attribute extraction". Bump the suffix if the note
        # text is meaningfully revised so older callers don't suppress the new copy.
        sentinel = '<<graphiti.attr_extraction.preamble.v1>>'
        target = messages[0]
        if sentinel in target.content:
            return
        if target.role == 'system':
            target.content += ATTRIBUTE_EXTRACTION_PREAMBLE
        else:
            target.content = ATTRIBUTE_EXTRACTION_PREAMBLE.lstrip() + '\n\n' + target.content

    def _apply_shared_prompt_parts(
        self,
        messages: list[Message],
        response_model: type[BaseModel] | None,
        attribute_extraction: bool,
        group_id: str | None,
        *,
        add_response_format: bool,
    ) -> tuple[str, ...]:
        shared_parts: list[str] = []
        if attribute_extraction and messages:
            self._apply_attribute_extraction_preamble(messages, True)
            shared_parts.append(ATTRIBUTE_EXTRACTION_PREAMBLE)

        if response_model is not None and add_response_format:
            serialized_model = json.dumps(response_model.model_json_schema())
            messages[-1].content += RESPONSE_FORMAT_TEMPLATE.replace(
                '{serialized_model}', serialized_model
            )
            shared_parts.append(RESPONSE_FORMAT_TEMPLATE)

        if messages:
            messages[0].content += get_extraction_language_instruction(group_id)
            shared_parts.append(LANGUAGE_INSTRUCTION_TEMPLATE)
        return tuple(shared_parts)

    @staticmethod
    def _prompt_version(
        prompt_template: PromptTemplate | None, shared_parts: tuple[str, ...]
    ) -> str:
        return (
            prompt_template.version(shared_parts)
            if prompt_template is not None
            else UNKNOWN_PROMPT_VERSION
        )

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
        prompt_template: PromptTemplate | None = None,
    ) -> dict[str, typing.Any]:
        if max_tokens is None:
            max_tokens = self.max_tokens

        shared_parts = self._apply_shared_prompt_parts(
            messages,
            response_model,
            attribute_extraction,
            group_id,
            add_response_format=True,
        )
        prompt_version = self._prompt_version(prompt_template, shared_parts)

        for message in messages:
            message.content = self._clean_input(message.content)

        # Wrap entire operation in tracing span
        with self.tracer.start_span('llm.generate') as span:
            attributes = {
                'llm.provider': self._get_provider_type(),
                'model.size': model_size.value,
                'max_tokens': max_tokens,
                'cache.enabled': self.cache_enabled,
            }
            if prompt_name:
                attributes['prompt.name'] = prompt_name
            attributes['prompt.version'] = prompt_version
            span.add_attributes(attributes)

            # Check cache first
            if self.cache_enabled and self.cache_dir is not None:
                cache_key = self._get_cache_key(messages, model=model)
                cached_response = self.cache_dir.get(cache_key)
                if cached_response is not None:
                    logger.debug(f'Cache hit for {cache_key}')
                    span.add_attributes({'cache.hit': True})
                    return cached_response

            span.add_attributes({'cache.hit': False})

            # Execute LLM call. Expose prompt_name to downstream
            # _generate_response implementations via a task-local so they can
            # tag it on the request (e.g. for the inference tap).
            token = current_prompt_name.set(prompt_name)
            version_token = current_prompt_version.set(prompt_version)
            try:
                model_kwargs = {} if model is None else {'model': model}
                response = await self._generate_response_with_retry(
                    messages,
                    response_model,
                    max_tokens,
                    model_size,
                    **model_kwargs,
                )
            except Exception as e:
                span.set_status('error', str(e))
                span.record_exception(e)
                raise
            finally:
                current_prompt_name.reset(token)
                current_prompt_version.reset(version_token)

            # Cache response if enabled
            if self.cache_enabled and self.cache_dir is not None:
                cache_key = self._get_cache_key(messages, model=model)
                self.cache_dir.set(cache_key, response)

            return response

    def _get_provider_type(self) -> str:
        """Get provider type from class name."""
        class_name = self.__class__.__name__.lower()
        if 'openai' in class_name:
            return 'openai'
        elif 'anthropic' in class_name:
            return 'anthropic'
        elif 'gemini' in class_name:
            return 'gemini'
        elif 'groq' in class_name:
            return 'groq'
        else:
            return 'unknown'

    def _get_failed_generation_log(self, messages: list[Message], output: str | None) -> str:
        """
        Log structural metadata and truncated raw output for debugging failed
        generations, without including full message content that may contain PII.
        """
        log = f'Input messages: {len(messages)} message(s), '
        log += f'roles: {[m.role for m in messages]}\n'
        if output is not None:
            truncated = output[:500] + '...' if len(output) > 500 else output
            log += f'Raw output (truncated): {truncated}\n'
        else:
            log += 'No raw output available'
        return log
