"""Timing wrappers for LLM and embedder clients used by the MCP server."""

import logging
from time import perf_counter
from typing import Any

from graphiti_core.embedder.client import EmbedderClient
from graphiti_core.llm_client import LLMClient


logger = logging.getLogger(__name__)


class InstrumentedLLMClient(LLMClient):
    """LLM wrapper that preserves isinstance checks while logging prompt latency."""

    def __init__(self, inner: LLMClient):
        super().__init__(inner.config, cache=False)
        self._inner = inner
        self.model = inner.model
        self.small_model = inner.small_model
        self.temperature = inner.temperature
        self.max_tokens = inner.max_tokens
        self.tracer = inner.tracer
        self.token_tracker = inner.token_tracker

    def __getattr__(self, name: str) -> Any:
        return getattr(self._inner, name)

    def set_tracer(self, tracer) -> None:
        self.tracer = tracer
        self._inner.set_tracer(tracer)

    async def _generate_response(self, *args, **kwargs):
        return await self._inner._generate_response(*args, **kwargs)

    async def generate_response(self, *args, **kwargs):
        prompt_name = kwargs.get('prompt_name') or 'unknown'
        model_size = kwargs.get('model_size')
        model_size_value = getattr(model_size, 'value', model_size) or 'unknown'
        model_name = getattr(self._inner, 'model', None) or 'unknown'

        started_at = perf_counter()
        try:
            return await self._inner.generate_response(*args, **kwargs)
        finally:
            elapsed_ms = (perf_counter() - started_at) * 1000
            logger.info(
                'LLM timing prompt=%s model=%s size=%s elapsed_ms=%.1f wrapper=%s client=%s',
                prompt_name,
                model_name,
                model_size_value,
                elapsed_ms,
                self.__class__.__name__,
                self._inner.__class__.__name__,
            )


class InstrumentedEmbedderClient(EmbedderClient):
    """Embedder wrapper that preserves isinstance checks while logging latency."""

    def __init__(self, inner: EmbedderClient, model_name: str, dimensions: int):
        self._inner = inner
        self._model_name = model_name
        self._dimensions = dimensions

    def __getattr__(self, name: str) -> Any:
        return getattr(self._inner, name)

    async def create(self, input_data):
        started_at = perf_counter()
        try:
            return await self._inner.create(input_data)
        finally:
            elapsed_ms = (perf_counter() - started_at) * 1000
            logger.info(
                'Embedder timing model=%s dimensions=%s batch_size=1 elapsed_ms=%.1f wrapper=%s client=%s',
                self._model_name,
                self._dimensions,
                elapsed_ms,
                self.__class__.__name__,
                self._inner.__class__.__name__,
            )

    async def create_batch(self, input_data_list):
        started_at = perf_counter()
        try:
            return await self._inner.create_batch(input_data_list)
        finally:
            elapsed_ms = (perf_counter() - started_at) * 1000
            logger.info(
                'Embedder timing model=%s dimensions=%s batch_size=%s elapsed_ms=%.1f wrapper=%s client=%s',
                self._model_name,
                self._dimensions,
                len(input_data_list),
                elapsed_ms,
                self.__class__.__name__,
                self._inner.__class__.__name__,
            )
