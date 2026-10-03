"""
Claude Code SDK LLM Client for Graphiti.

Uses the Claude Agent SDK to make LLM calls through a persistent subprocess.
Authenticates via Claude Code's OAuth credentials (~/.claude) — no API key needed.

The client keeps one subprocess alive and monitors context usage. When context
exceeds the threshold, it reconnects (kills + restarts the subprocess).
"""

import asyncio
import contextlib
import json
import logging
import typing

from pydantic import BaseModel

from ..prompts.models import Message
from .client import LLMClient
from .config import LLMConfig, ModelSize

logger = logging.getLogger(__name__)

DEFAULT_MODEL = 'sonnet'
DEFAULT_MAX_TOKENS = 16384
CONTEXT_RESET_THRESHOLD = 60  # percentage — reconnect when context exceeds this


class ClaudeCodeClient(LLMClient):
    """LLM client that uses Claude Code SDK (subprocess) for inference.

    Keeps a persistent ClaudeSDKClient subprocess. Each _generate_response call
    sends a query and parses JSON from the response text. Context is monitored
    and the subprocess is recycled when it gets too full.
    """

    def __init__(
        self,
        config: LLMConfig | None = None,
        cache: bool = False,
        context_reset_pct: int = CONTEXT_RESET_THRESHOLD,
    ):
        if config is None:
            config = LLMConfig(model=DEFAULT_MODEL, max_tokens=DEFAULT_MAX_TOKENS)
        if config.model is None:
            config.model = DEFAULT_MODEL
        super().__init__(config, cache)

        self._context_reset_pct = context_reset_pct
        self._client = None
        self._call_count = 0
        self._lock = asyncio.Lock()

    async def _ensure_client(self):
        """Create or reuse the Claude SDK client."""
        if self._client is None:
            from claude_agent_sdk import ClaudeAgentOptions, ClaudeSDKClient

            options = ClaudeAgentOptions(
                system_prompt='You are a JSON extraction engine. Respond with ONLY valid JSON. No markdown fences, no explanation, no commentary.',
                model=self.model or DEFAULT_MODEL,
                max_turns=1,
                permission_mode='plan',
            )
            self._client = ClaudeSDKClient(options=options)
            await self._client.connect()
            self._call_count = 0
            logger.info('Claude Code subprocess started')

    async def _check_context_and_reset(self):
        """Check context usage and reset if needed."""
        if self._client is None:
            return

        self._call_count += 1

        # Check every 5 calls to avoid overhead
        if self._call_count % 5 != 0:
            return

        try:
            usage = await self._client.get_context_usage()
            pct = usage.get('percentage', 0)
            if pct >= self._context_reset_pct:
                logger.info(
                    f'Context at {pct}% (threshold {self._context_reset_pct}%) — resetting subprocess'
                )
                await self._reset_client()
        except Exception as e:
            logger.debug(f'Context check failed: {e}')

    async def _reset_client(self):
        """Kill and restart the subprocess."""
        if self._client:
            with contextlib.suppress(Exception):
                await self._client.disconnect()
            self._client = None
        await self._ensure_client()

    async def close(self):
        """Clean up the subprocess."""
        if self._client:
            with contextlib.suppress(Exception):
                await self._client.disconnect()
            self._client = None

    async def _generate_response(
        self,
        messages: list[Message],
        response_model: type[BaseModel] | None = None,
        max_tokens: int = DEFAULT_MAX_TOKENS,
        model_size: ModelSize = ModelSize.medium,
    ) -> dict[str, typing.Any]:
        async with self._lock:
            await self._ensure_client()

            # Build prompt from messages
            prompt_parts = []
            for m in messages:
                if m.role == 'system':
                    prompt_parts.append(f'[System]: {m.content}')
                elif m.role == 'user':
                    prompt_parts.append(m.content)

            prompt = '\n\n'.join(prompt_parts)

            try:
                from claude_agent_sdk import ResultMessage

                await self._client.query(prompt)

                result_text = ''
                async for msg in self._client.receive_response():
                    if hasattr(msg, 'result') and msg.result:
                        result_text = msg.result
                    elif hasattr(msg, 'text') and msg.text:
                        result_text = msg.text
                    if isinstance(msg, ResultMessage):
                        break

                await self._check_context_and_reset()

                # Parse JSON from response
                return self._extract_json(result_text)

            except Exception as e:
                logger.error(f'Claude Code SDK error: {e}')
                # Reset on error — subprocess might be in bad state
                await self._reset_client()
                raise

    def _extract_json(self, text: str) -> dict[str, typing.Any]:
        """Extract JSON from response text."""
        # Try direct parse first
        try:
            return json.loads(text)
        except (json.JSONDecodeError, TypeError):
            pass

        # Find JSON in text (may have markdown fences or explanation)
        json_start = text.find('{')
        json_end = text.rfind('}') + 1
        if json_start >= 0 and json_end > json_start:
            try:
                return json.loads(text[json_start:json_end])
            except json.JSONDecodeError:
                pass

        # Try array
        json_start = text.find('[')
        json_end = text.rfind(']') + 1
        if json_start >= 0 and json_end > json_start:
            try:
                return {'items': json.loads(text[json_start:json_end])}
            except json.JSONDecodeError:
                pass

        raise ValueError(f'Could not extract JSON from response: {text[:200]}')
