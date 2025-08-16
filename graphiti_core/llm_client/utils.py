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

import logging
from time import time

from graphiti_core.embedder.client import EmbedderClient

from .config import DEFAULT_MAX_TOKENS

logger = logging.getLogger(__name__)


def resolve_max_tokens(
    requested_max_tokens: int | None = None,
    config_max_tokens: int | None = None,
    instance_max_tokens: int | None = None,
    default_max_tokens: int = DEFAULT_MAX_TOKENS,
) -> int:
    """
    Resolve the maximum output tokens to use based on precedence rules.
    
    Precedence order (highest to lowest):
    1. Explicit max_tokens parameter passed to generate_response()
    2. Config max_tokens set in LLMConfig (if different from default)
    3. Instance max_tokens set during client initialization
    4. Default max_tokens as final fallback

    Args:
        requested_max_tokens: The max_tokens parameter passed to generate_response()
        config_max_tokens: The max_tokens from LLMConfig
        instance_max_tokens: The max_tokens set during client initialization
        default_max_tokens: The default fallback value for the client and model

    Returns:
        int: The resolved maximum tokens to use
    """
    # 1. Use explicit parameter if provided
    if requested_max_tokens is not None:
        return requested_max_tokens

    # 2. Use config max_tokens if explicitly set (different from default)
    if config_max_tokens is not None and config_max_tokens != default_max_tokens:
        return config_max_tokens

    # 3. Use instance max_tokens if set during initialization
    if instance_max_tokens is not None:
        return instance_max_tokens
        
    # 4. Use default as final fallback
    return default_max_tokens


async def generate_embedding(embedder: EmbedderClient, text: str):
    start = time()

    text = text.replace('\n', ' ')
    embedding = await embedder.create(input_data=[text])

    end = time()
    logger.debug(f'embedded text of length {len(text)} in {end - start} ms')

    return embedding
