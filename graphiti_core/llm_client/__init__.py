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

from .client import LLMClient
from .config import LLMConfig
from .errors import RateLimitError
from .openai_client import OpenAIClient
from .openai_generic_client import OpenAIGenericClient

__all__ = [
    'LLMClient',
    'OpenAIClient',
    'OpenAIGenericClient',
    'LLMConfig',
    'RateLimitError',
]

# Optional: AnthropicClient
try:
    from .anthropic_client import AnthropicClient
    __all__.append('AnthropicClient')
except ImportError:
    AnthropicClient = None  # type: ignore

# Optional: GeminiClient
try:
    from .gemini_client import GeminiClient
    __all__.append('GeminiClient')
except ImportError:
    GeminiClient = None  # type: ignore

# Optional: GroqClient
try:
    from .groq_client import GroqClient
    __all__.append('GroqClient')
except ImportError:
    GroqClient = None  # type: ignore
