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

import json
import logging
import boto3
from typing import ClassVar
from pydantic import BaseModel

from .config import DEFAULT_MAX_TOKENS, LLMConfig
from .openai_base_client import BaseOpenAIClient

logger = logging.getLogger(__name__)

# bedrock support :) cuz i really wanted it.

class BedrockAnthropicLLMClient(BaseOpenAIClient):
    MAX_RETRIES: ClassVar[int] = 2

    def __init__(
        self,
        config: LLMConfig | None = None,
        max_tokens: int = DEFAULT_MAX_TOKENS,
    ):
        super().__init__(config, cache=False, max_tokens=max_tokens)

        self.region = config.region if config and getattr(config, "region", None) else "us-east-1"
        self.model = config.model if config else "anthropic.claude-3-sonnet-20240229-v1:0"

        self.client = boto3.client("bedrock-runtime", region_name=self.region)
    async def _create_structured_completion(
        self,
        model: str,
        messages: list[dict],
        temperature: float | None,
        max_tokens: int,
        response_model: type[BaseModel],
    ):
       
        text_response = await self._invoke_bedrock_model(
            model=model,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
            response_format="json",
        )

        try:
            parsed = response_model.parse_raw(text_response)
            return parsed
        except Exception as e:
            logger.error(f"Failed to parse structured Bedrock response: {e}")
            raise

    async def _create_completion(
        self,
        model: str,
        messages: list[dict],
        temperature: float | None,
        max_tokens: int,
        response_model: type[BaseModel] | None = None,
    ):
        text_response = await self._invoke_bedrock_model(
            model=model,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
            response_format="text",
        )

        if response_model:
            try:
                parsed = response_model.parse_raw(text_response)
                return parsed
            except Exception as e:
                logger.error(f"Failed to parse Bedrock response to {response_model.__name__}: {e}")
                raise
        return text_response

    async def _invoke_bedrock_model(
        self,
        model: str,
        messages: list[dict],
        temperature: float | None = 0.7,
        max_tokens: int = 200,
        response_format: str = "text",
    ) -> str:

        # Separate system prompt and user messages
        system_prompt = None
        final_messages = []
        for m in messages:
            if m["role"] == "system":
                system_prompt = m["content"]
            else:
                final_messages.append(m)

        body_dict = {
            "messages": final_messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "anthropic_version": "bedrock-2023-05-31",
        }

        if system_prompt:
            body_dict["system"] = system_prompt

        body = json.dumps(body_dict)

        try:
            resp = self.client.invoke_model(
                modelId=model,
                body=body,
                accept="application/json",
                contentType="application/json",
            )
            data = json.loads(resp["body"].read().decode("utf-8"))

            if "content" in data and len(data["content"]) > 0:
                text = data["content"][0].get("text", "")
            elif "outputText" in data:
                text = data["outputText"]
            else:
                text = json.dumps(data)

            # Ensure JSON-object-only for structured calls
            if response_format == "json":
                text = text.strip()
                if text.startswith("```json"):
                    text = text.replace("```json", "").replace("```", "").strip()
            return text.strip()

        except Exception as e:
            logger.error(f"Bedrock model invocation failed: {e}", exc_info=True)
            raise
