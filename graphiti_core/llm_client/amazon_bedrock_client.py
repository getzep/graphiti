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

import asyncio
import json
import logging
from typing import TYPE_CHECKING

from pydantic import BaseModel

from ..prompts.models import Message
from .client import LLMClient
from .config import DEFAULT_MAX_TOKENS, LLMConfig
from .errors import RateLimitError

if TYPE_CHECKING:
    import boto3
else:
    try:
        import boto3
    except ImportError:
        raise ImportError(
            'boto3 is required for AmazonBedrockLLMClient. '
            'Install it with: pip install graphiti-core[bedrock]'
        ) from None

logger = logging.getLogger(__name__)


class AmazonBedrockLLMClient(LLMClient):
    def __init__(
        self,
        config: LLMConfig | None = None,
        max_tokens: int = DEFAULT_MAX_TOKENS,
        region: str = 'us-east-1',
    ):
        super().__init__(config, cache=False)

        self.region = region
        self.model = (
            config.model
            if config and config.model
            else 'us.anthropic.claude-sonnet-4-20250514-v1:0'
        )
        self.temperature = config.temperature if config and config.temperature is not None else 0.7
        self.max_tokens = max_tokens

        self.client = boto3.client('bedrock-runtime', region_name=self.region)

    async def _generate_response(
        self,
        messages: list[Message],
        response_model: type[BaseModel] | None = None,
        max_tokens: int = DEFAULT_MAX_TOKENS,
        model_size=None,
    ):
        # Convert Message objects to dict format
        message_dicts = [msg.model_dump() for msg in messages]

        if response_model:
            # Add JSON schema instruction for structured output
            schema = response_model.model_json_schema()
            schema_str = json.dumps(schema, indent=2)

            # Modify the last user message to include schema instruction
            if message_dicts and message_dicts[-1]['role'] == 'user':
                message_dicts[-1]['content'] += (
                    f'\n\nPlease respond with valid JSON that follows this exact schema:\n{schema_str}\n\nIMPORTANT: Return ONLY valid JSON with no extra text, explanations, or markdown formatting.'
                )

            text_response = await self._invoke_bedrock_model(
                model=self.model,
                messages=message_dicts,
                temperature=self.temperature,
                max_tokens=max_tokens,
                response_format='json',
            )

            try:
                parsed_model = response_model.model_validate_json(text_response)
                return parsed_model.model_dump()
            except Exception as e:
                logger.error(f'Failed to parse structured Bedrock response: {e}')
                logger.error(f'Raw response: {text_response}')
                raise
        else:
            text_response = await self._invoke_bedrock_model(
                model=self.model,
                messages=message_dicts,
                temperature=self.temperature,
                max_tokens=max_tokens,
                response_format='text',
            )
            return {'content': text_response}

    async def _invoke_bedrock_model(
        self,
        model: str,
        messages: list[dict],
        temperature: float,
        max_tokens: int,
        response_format: str,
    ) -> str:
        # Separate system prompt and user messages
        system_prompt = None
        final_messages = [m for m in messages if m['role'] != 'system']

        for m in messages:
            if m['role'] == 'system':
                system_prompt = m['content']
                break

        body_dict = {
            'messages': final_messages,
            'temperature': temperature,
            'max_tokens': max_tokens,
            'anthropic_version': 'bedrock-2023-05-31',
        }

        if system_prompt:
            body_dict['system'] = system_prompt

        body = json.dumps(body_dict)

        try:
            # Use executor to run sync boto3 call in async context
            loop = asyncio.get_event_loop()
            resp = await loop.run_in_executor(
                None,
                lambda: self.client.invoke_model(
                    modelId=model,
                    body=body,
                    accept='application/json',
                    contentType='application/json',
                ),
            )

            data = json.loads(resp['body'].read().decode('utf-8'))

            if 'content' in data and data['content']:
                text = data['content'][0].get('text', '')
            elif 'outputText' in data:
                text = data['outputText']
            else:
                text = json.dumps(data)

            # Clean JSON response format
            if response_format == 'json':
                text = self._clean_json_response(text)
            return text.strip()

        except Exception as e:
            if 'throttling' in str(e).lower() or 'rate' in str(e).lower():
                raise RateLimitError(f'Rate limit exceeded: {e}') from e
            logger.error(f'Bedrock model invocation failed: {e}')
            raise

    def _clean_json_response(self, text: str) -> str:
        """Clean JSON response from markdown formatting and extract JSON."""
        import re

        text = text.strip()

        logger.debug(f'Raw Bedrock response: {text[:500]}...')

        # Remove code blocks
        text = re.sub(r'^```(?:json|JSON)?\s*', '', text, flags=re.MULTILINE)
        text = re.sub(r'```\s*$', '', text, flags=re.MULTILINE)

        # Fix double braces issue
        text = re.sub(r'^\{\{', '{', text)
        text = re.sub(r'\}\}$', '}', text)

        # Find JSON object - look for first { and last }
        start_idx = text.find('{')
        if start_idx != -1:
            # Find the matching closing brace
            brace_count = 0
            end_idx = -1
            for i in range(start_idx, len(text)):
                if text[i] == '{':
                    brace_count += 1
                elif text[i] == '}':
                    brace_count -= 1
                    if brace_count == 0:
                        end_idx = i + 1
                        break

            if end_idx != -1:
                text = text[start_idx:end_idx]

        logger.debug(f'Cleaned JSON response: {text[:200]}...')
        return text.strip()
