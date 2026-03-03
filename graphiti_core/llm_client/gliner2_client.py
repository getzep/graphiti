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

import ast
import asyncio
import json
import logging
import re
import typing
from time import perf_counter
from typing import TYPE_CHECKING

from pydantic import BaseModel

from ..prompts.models import Message
from .client import LLMClient
from .config import DEFAULT_MAX_TOKENS, LLMConfig, ModelSize
from .errors import RateLimitError

if TYPE_CHECKING:
    from gliner2 import GLiNER2  # type: ignore[import-untyped]
else:
    try:
        from gliner2 import GLiNER2  # type: ignore[import-untyped]
    except ImportError:
        raise ImportError(
            'gliner2 is required for GLiNER2Client. '
            'Install it with: pip install graphiti-core[gliner2]'
        ) from None

logger = logging.getLogger(__name__)

DEFAULT_MODEL = 'fastino/gliner2-base-v1'
DEFAULT_THRESHOLD = 0.5

# Response model that GLiNER2 handles natively
_ENTITY_EXTRACTION_MODEL = 'ExtractedEntities'


class GLiNER2Client(LLMClient):
    """LLM client that uses GLiNER2 for entity extraction.

    GLiNER2 is a lightweight extraction model (205M-340M params) that handles
    named entity recognition locally on CPU. All other operations (edge/relation
    extraction, deduplication, summarization, etc.) are delegated to the
    required llm_client.

    Note: When using local models (no base_url), initialization loads model
    weights synchronously. Create this client before entering the async
    event loop (e.g., before ``asyncio.run()``).
    """

    def __init__(
        self,
        config: LLMConfig | None = None,
        cache: bool = False,
        threshold: float = DEFAULT_THRESHOLD,
        include_confidence: bool = False,
        llm_client: LLMClient | None = None,
    ) -> None:
        if llm_client is None:
            raise ValueError(
                'llm_client is required. GLiNER2 cannot handle all operations '
                '(deduplication, summarization, etc.) and must delegate to a '
                'general-purpose LLM client.'
            )

        if config is None:
            config = LLMConfig()

        super().__init__(config, cache)

        self.threshold = threshold
        self.include_confidence = include_confidence
        self.llm_client = llm_client
        self.extraction_latencies: list[float] = []

        model_id = config.model or DEFAULT_MODEL
        small_model_id = config.small_model or model_id

        if config.base_url:
            logger.info('Initializing GLiNER2 in API mode: %s', config.base_url)
            self._model = GLiNER2.from_api(
                api_key=config.api_key or '',
                api_base_url=config.base_url,
            )
            self._small_model = self._model
        else:
            logger.info('Loading GLiNER2 model: %s', model_id)
            self._model = GLiNER2.from_pretrained(model_id)
            if small_model_id != model_id:
                logger.info('Loading GLiNER2 small model: %s', small_model_id)
                self._small_model = GLiNER2.from_pretrained(small_model_id)
            else:
                self._small_model = self._model

    def _get_model_for_size(self, model_size: ModelSize) -> typing.Any:
        if model_size == ModelSize.small:
            return self._small_model
        return self._model

    def _get_provider_type(self) -> str:
        return 'gliner2'

    # ── Message parsing helpers ──────────────────────────────────────

    @staticmethod
    def _extract_text_from_messages(messages: list[Message]) -> str:
        """Extract the raw text content from the message list for GLiNER2 processing."""
        user_content = messages[-1].content if len(messages) > 1 else messages[0].content

        # Try known XML tags in priority order
        for tag in [
            'CURRENT MESSAGE',
            'CURRENT_MESSAGE',
            'TEXT',
            'JSON',
        ]:
            pattern = rf'<{re.escape(tag)}>\s*(.*?)\s*</{re.escape(tag)}>'
            match = re.search(pattern, user_content, re.DOTALL)
            if match:
                return match.group(1).strip()

        # Fallback: return the full user content
        return user_content

    @staticmethod
    def _extract_entity_labels(messages: list[Message]) -> tuple[dict[str, str], dict[str, int]]:
        """Extract entity type labels and id mappings from the message.

        Returns:
            Tuple of (labels_dict, label_to_id) where labels_dict maps
            entity_type_name → entity_type_description and label_to_id maps
            entity_type_name → entity_type_id.
        """
        user_content = messages[-1].content if len(messages) > 1 else messages[0].content

        match = re.search(
            r'<ENTITY TYPES>\s*(.*?)\s*</ENTITY TYPES>', user_content, re.DOTALL
        )
        if match:
            try:
                raw = match.group(1)
                # Prompt templates interpolate Python list[dict] directly,
                # producing Python repr (single quotes, None) rather than JSON.
                try:
                    entity_types = json.loads(raw)
                except json.JSONDecodeError:
                    entity_types = ast.literal_eval(raw)

                labels_dict: dict[str, str] = {}
                label_to_id: dict[str, int] = {}
                for et in entity_types:
                    name = et['entity_type_name']
                    labels_dict[name] = et.get('entity_type_description') or ''
                    label_to_id[name] = et['entity_type_id']
                return labels_dict, label_to_id
            except (json.JSONDecodeError, KeyError, ValueError, SyntaxError):
                logger.warning('Failed to parse <ENTITY TYPES> from message')

        return {'Entity': 'General entity'}, {'Entity': 0}

    # ── Extraction handlers ──────────────────────────────────────────

    async def _handle_entity_extraction(
        self,
        model: typing.Any,
        text: str,
        messages: list[Message],
    ) -> dict[str, typing.Any]:
        """Handle entity extraction using GLiNER2.

        Maps GLiNER2 output format to Graphiti's ExtractedEntities format.
        """
        labels_dict, label_to_id = self._extract_entity_labels(messages)

        result = await asyncio.to_thread(
            model.extract_entities,
            text,
            labels_dict,
            threshold=self.threshold,
            include_confidence=self.include_confidence,
        )

        extracted_entities: list[dict[str, typing.Any]] = []
        entities_dict = result.get('entities', {})

        for entity_type, entity_items in entities_dict.items():
            entity_type_id = label_to_id.get(entity_type, 0)
            for item in entity_items:
                # GLiNER2 returns strings or dicts (when include_confidence=True)
                name = item.get('text', '') if isinstance(item, dict) else str(item)

                if name:
                    extracted_entities.append({
                        'name': name,
                        'entity_type_id': entity_type_id,
                    })

        return {'extracted_entities': extracted_entities}

    # ── Core dispatch ────────────────────────────────────────────────

    def _is_gliner2_operation(self, response_model: type[BaseModel] | None) -> bool:
        """Determine if the response_model maps to a GLiNER2-native operation."""
        if response_model is None:
            return False
        return response_model.__name__ == _ENTITY_EXTRACTION_MODEL

    async def _generate_response(
        self,
        messages: list[Message],
        response_model: type[BaseModel] | None = None,
        max_tokens: int = DEFAULT_MAX_TOKENS,
        model_size: ModelSize = ModelSize.medium,
    ) -> dict[str, typing.Any]:
        model = self._get_model_for_size(model_size)
        text = self._extract_text_from_messages(messages)

        if not text:
            logger.warning('No text extracted from messages for GLiNER2 processing')
            return {'extracted_entities': []}

        try:
            t0 = perf_counter()
            result = await self._handle_entity_extraction(model, text, messages)
            latency_ms = (perf_counter() - t0) * 1000
            self.extraction_latencies.append(latency_ms)
            logger.info('GLiNER2 entity extraction: %.1f ms', latency_ms)
            return result
        except Exception as e:
            error_msg = str(e).lower()
            if 'rate limit' in error_msg or '429' in error_msg:
                raise RateLimitError(f'GLiNER2 API rate limit: {e}') from e
            if 'authentication' in error_msg or 'unauthorized' in error_msg:
                raise
            logger.error('GLiNER2 extraction error: %s', e)
            raise

    async def generate_response(
        self,
        messages: list[Message],
        response_model: type[BaseModel] | None = None,
        max_tokens: int | None = None,
        model_size: ModelSize = ModelSize.medium,
        group_id: str | None = None,
        prompt_name: str | None = None,
    ) -> dict[str, typing.Any]:
        # Delegate non-extraction operations to the LLM client
        if not self._is_gliner2_operation(response_model):
            return await self.llm_client.generate_response(
                messages,
                response_model=response_model,
                max_tokens=max_tokens,
                model_size=model_size,
                group_id=group_id,
                prompt_name=prompt_name,
            )

        if max_tokens is None:
            max_tokens = self.max_tokens

        # Clean input (still useful for the text we extract)
        for message in messages:
            message.content = self._clean_input(message.content)

        with self.tracer.start_span('llm.generate') as span:
            attributes: dict[str, typing.Any] = {
                'llm.provider': 'gliner2',
                'model.size': model_size.value,
                'cache.enabled': self.cache_enabled,
            }
            if prompt_name:
                attributes['prompt.name'] = prompt_name
            span.add_attributes(attributes)

            # Check cache
            if self.cache_enabled and self.cache_dir is not None:
                cache_key = self._get_cache_key(messages)
                cached_response = self.cache_dir.get(cache_key)
                if cached_response is not None:
                    logger.debug('Cache hit for %s', cache_key)
                    span.add_attributes({'cache.hit': True})
                    return cached_response

            span.add_attributes({'cache.hit': False})

            try:
                response = await self._generate_response_with_retry(
                    messages, response_model, max_tokens, model_size
                )

                # Approximate token usage (GLiNER2 doesn't report actual tokens)
                text = self._extract_text_from_messages(messages)
                input_tokens = len(text) // 4
                output_tokens = len(json.dumps(response)) // 4
                self.token_tracker.record(
                    prompt_name or 'unknown',
                    input_tokens,
                    output_tokens,
                )
            except Exception as e:
                span.set_status('error', str(e))
                span.record_exception(e)
                raise

            # Cache response
            if self.cache_enabled and self.cache_dir is not None:
                cache_key = self._get_cache_key(messages)
                self.cache_dir.set(cache_key, response)

            return response
