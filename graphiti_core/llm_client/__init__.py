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
