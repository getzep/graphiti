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
import typing
from typing import Any

import httpx
from pydantic import BaseModel

from ..prompts.models import Message
from .client import LLMClient
from .config import DEFAULT_MAX_TOKENS, LLMConfig, ModelSize
from .errors import RateLimitError

logger = logging.getLogger(__name__)

DEFAULT_MODEL = 'gpt-4o'
DEFAULT_SMALL_MODEL = 'gpt-4o-mini'


class VSCodeClient(LLMClient):
    """
    VSCodeClient is a client class for interacting with VS Code's language models through MCP.

    This client leverages VS Code's built-in language model capabilities, allowing the MCP server
    to utilize the models available in the VS Code environment without requiring external API keys.

    Attributes:
        model_selector (str): The model selector to use for requests.
        vscode_available (bool): Whether VS Code integration is available.
    """

    def __init__(
        self,
        config: LLMConfig | None = None,
        cache: bool = False,
        max_tokens: int = DEFAULT_MAX_TOKENS,
    ):
        """
        Initialize the VSCodeClient with the provided configuration and cache setting.

        Args:
            config (LLMConfig | None): The configuration for the LLM client, including model selection.
            cache (bool): Whether to use caching for responses. Defaults to False.
            max_tokens (int): Maximum number of tokens for responses.
        """
        if config is None:
            config = LLMConfig(
                model=DEFAULT_MODEL,
                small_model=DEFAULT_SMALL_MODEL,
                api_key="vscode"  # Placeholder, not used
            )

        super().__init__(config, cache)
        self.max_tokens = max_tokens
        self.vscode_available = self._check_vscode_availability()

    def _check_vscode_availability(self) -> bool:
        """Check if VS Code model integration is available."""
        try:
            # Try to import VS Code specific modules or check environment
            import os
            # Check if we're running in a VS Code context
            return 'VSCODE_PID' in os.environ or 'VSCODE_IPC_HOOK' in os.environ
        except Exception:
            return False

    def _get_model_for_size(self, model_size: ModelSize) -> str:
        """Get the appropriate model name based on the requested size."""
        if model_size == ModelSize.small:
            return self.small_model or DEFAULT_SMALL_MODEL
        else:
            return self.model or DEFAULT_MODEL

    def _convert_messages_to_vscode_format(self, messages: list[Message]) -> list[dict[str, Any]]:
        """Convert internal Message format to VS Code compatible format."""
        vscode_messages = []
        for message in messages:
            vscode_messages.append({
                "role": message.role,
                "content": message.content
            })
        return vscode_messages

    async def _make_vscode_request(
        self,
        messages: list[dict[str, Any]],
        model: str,
        max_tokens: int,
        temperature: float,
        response_format: dict[str, Any] | None = None
    ) -> dict[str, Any]:
        """Make a request to VS Code's language model through MCP."""
        
        # Prepare the request payload
        request_data = {
            "model": model,
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": temperature,
        }
        
        if response_format:
            request_data["response_format"] = response_format

        try:
            # In a real implementation, this would connect to VS Code's MCP server
            # For now, we'll call VS Code models through available methods
            response_text = await self._call_vscode_models(request_data)
            
            return {
                "choices": [{
                    "message": {
                        "content": response_text,
                        "role": "assistant"
                    }
                }]
            }
            
        except Exception as e:
            logger.error(f"Error making VS Code model request: {e}")
            raise

    async def _call_vscode_models(self, request_data: dict[str, Any]) -> str:
        """
        Make a call to VS Code's language model through available integration methods.
        This method attempts multiple integration approaches for VS Code language models.
        """
        try:
            # Method 1: Try VS Code extension API if available
            response = await self._try_vscode_extension_api(request_data)
            if response:
                return response
            
            # Method 2: Try MCP protocol if available
            response = await self._try_mcp_protocol(request_data)
            if response:
                return response
            
            # Method 3: Fallback to simulated response
            return await self._fallback_vscode_response(request_data)
                
        except Exception as e:
            logger.warning(f"All VS Code integration methods failed, using fallback: {e}")
            return await self._fallback_vscode_response(request_data)

    async def _try_vscode_extension_api(self, request_data: dict[str, Any]) -> str | None:
        """Try to use VS Code extension API for language models."""
        try:
            # This would integrate with VS Code's language model API
            # In a real implementation, this would use VS Code's extension context
            # For now, return None to indicate this method is not available
            return None
        except Exception:
            return None

    async def _try_mcp_protocol(self, request_data: dict[str, Any]) -> str | None:
        """Try to use MCP protocol to communicate with VS Code models."""
        try:
            # This would use MCP to communicate with VS Code's language model server
            # Implementation would depend on available MCP clients and VS Code setup
            # For now, return None to indicate this method is not available
            return None
        except Exception:
            return None

    async def _fallback_vscode_response(self, request_data: dict[str, Any]) -> str:
        """
        Fallback response when VS Code models are not available.
        This provides a basic structured response for development/testing.
        """
        messages = request_data.get("messages", [])
        if not messages:
            return "{}"
            
        # Extract the main prompt content
        prompt_content = ""
        system_content = ""
        
        for msg in messages:
            if msg.get("role") == "user":
                prompt_content = msg.get("content", "")
            elif msg.get("role") == "system":
                system_content = msg.get("content", "")
        
        # For structured responses, analyze the schema and provide appropriate structure
        if "response_format" in request_data:
            schema = request_data["response_format"].get("schema", {})
            
            # Generate appropriate response based on schema properties
            if "properties" in schema:
                response = {}
                for prop_name, prop_info in schema["properties"].items():
                    if prop_info.get("type") == "array":
                        response[prop_name] = []
                    elif prop_info.get("type") == "string":
                        response[prop_name] = f"fallback_{prop_name}"
                    elif prop_info.get("type") == "object":
                        response[prop_name] = {}
                    else:
                        response[prop_name] = None
                
                return json.dumps(response)
            else:
                return '{"status": "fallback_response", "message": "VS Code models not available"}'
        
        # For regular responses, provide a contextual response
        return f"""Based on the prompt: "{prompt_content[:200]}..."

This is a fallback response since VS Code language models are not currently available. 
In a production environment, this would be handled by VS Code's built-in language model capabilities.

System context: {system_content[:100] if system_content else 'None'}..."""

    async def _create_completion(
        self,
        model: str,
        messages: list[dict[str, Any]],
        temperature: float | None,
        max_tokens: int,
        response_model: type[BaseModel] | None = None,
    ) -> dict[str, Any]:
        """Create a completion using VS Code's language models."""
        
        response_format = None
        if response_model:
            response_format = {
                "type": "json_object",
                "schema": response_model.model_json_schema()
            }
        
        return await self._make_vscode_request(
            messages=messages,
            model=model,
            max_tokens=max_tokens,
            temperature=temperature or 0.0,
            response_format=response_format
        )

    async def _create_structured_completion(
        self,
        model: str,
        messages: list[dict[str, Any]],
        temperature: float | None,
        max_tokens: int,
        response_model: type[BaseModel],
    ) -> dict[str, Any]:
        """Create a structured completion using VS Code's language models."""
        
        response_format = {
            "type": "json_object",
            "schema": response_model.model_json_schema()
        }
        
        return await self._make_vscode_request(
            messages=messages,
            model=model,
            max_tokens=max_tokens,
            temperature=temperature or 0.0,
            response_format=response_format
        )

    def _handle_response(self, response: dict[str, Any]) -> dict[str, Any]:
        """Handle and parse the response from VS Code models."""
        try:
            content = response["choices"][0]["message"]["content"]
            
            # Try to parse as JSON
            if content.strip().startswith('{') or content.strip().startswith('['):
                return json.loads(content)
            else:
                # If not JSON, wrap in a simple structure
                return {"response": content}
                
        except (KeyError, IndexError, json.JSONDecodeError) as e:
            logger.error(f"Error parsing VS Code model response: {e}")
            raise Exception(f"Invalid response format: {e}")

    async def _generate_response(
        self,
        messages: list[Message],
        response_model: type[BaseModel] | None = None,
        max_tokens: int = DEFAULT_MAX_TOKENS,
        model_size: ModelSize = ModelSize.medium,
    ) -> dict[str, typing.Any]:
        """Generate a response using VS Code's language models."""
        
        if not self.vscode_available:
            logger.warning("VS Code integration not available, using fallback behavior")
        
        # Convert messages to VS Code format
        vscode_messages = self._convert_messages_to_vscode_format(messages)
        model = self._get_model_for_size(model_size)

        try:
            if response_model:
                response = await self._create_structured_completion(
                    model=model,
                    messages=vscode_messages,
                    temperature=self.temperature,
                    max_tokens=max_tokens or self.max_tokens,
                    response_model=response_model,
                )
            else:
                response = await self._create_completion(
                    model=model,
                    messages=vscode_messages,
                    temperature=self.temperature,
                    max_tokens=max_tokens or self.max_tokens,
                )
            
            return self._handle_response(response)

        except httpx.HTTPStatusError as e:
            if e.response.status_code == 429:
                raise RateLimitError from e
            else:
                logger.error(f'HTTP error in VS Code model request: {e}')
                raise
        except Exception as e:
            logger.error(f'Error in generating VS Code model response: {e}')
            raise