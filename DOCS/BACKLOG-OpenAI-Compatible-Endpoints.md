# OpenAI-Compatible Custom Endpoint Support in Graphiti

## Overview

This document analyzes how Graphiti handles OpenAI-compatible custom endpoints (like OpenRouter, NagaAI, Together.ai, etc.) and provides recommendations for improving support.

## Current Architecture

Graphiti has **three main OpenAI-compatible client implementations**:

### 1. OpenAIClient (Default)

**File**: `graphiti_core/llm_client/openai_client.py`

- Extends `BaseOpenAIClient`
- Uses the **new OpenAI Responses API** (`/v1/responses` endpoint)
- Uses `client.responses.parse()` for structured outputs (OpenAI SDK v1.91+)
- This is the **default client** exported in the public API

```python
response = await self.client.responses.parse(
    model=model,
    input=messages,
    temperature=temperature,
    max_output_tokens=max_tokens,
    text_format=response_model,
    reasoning={'effort': reasoning},
    text={'verbosity': verbosity},
)
```

### 2. OpenAIGenericClient (Legacy)

**File**: `graphiti_core/llm_client/openai_generic_client.py`

- Uses the **standard Chat Completions API** (`/v1/chat/completions`)
- Uses `client.chat.completions.create()` 
- **Only supports unstructured JSON responses** (not Pydantic schemas)
- Currently **not exported** in `__init__.py` (hidden from public API)

```python
response = await self.client.chat.completions.create(
    model=model,
    messages=messages,
    temperature=temperature,
    max_tokens=max_tokens,
    response_format={'type': 'json_object'},
)
```

### 3. AzureOpenAILLMClient

**File**: `graphiti_core/llm_client/azure_openai_client.py`

- Azure-specific implementation
- Also uses `responses.parse()` like `OpenAIClient`
- Handles Azure-specific authentication and endpoints

## The Root Problem

### Issue Description

When users configure Graphiti with custom OpenAI-compatible endpoints, they encounter errors because:

1. **`OpenAIClient` uses the new `/v1/responses` endpoint** via `client.responses.parse()`
   - This is a **new OpenAI API** (introduced in OpenAI SDK v1.91.0) for structured outputs
   - This endpoint is **proprietary to OpenAI** and **not part of the standard OpenAI-compatible API specification**
   
2. **Most OpenAI-compatible services** (OpenRouter, NagaAI, Ollama, Together.ai, etc.) **only implement** the standard `/v1/chat/completions` endpoint
   - They do **NOT** implement `/v1/responses`
   
3. When you configure a `base_url` pointing to these services, Graphiti tries to call:
   ```
   https://your-custom-endpoint.com/v1/responses
   ```
   Instead of the expected:
   ```
   https://your-custom-endpoint.com/v1/chat/completions
   ```

### Example Error Scenario

```python
from graphiti_core import Graphiti
from graphiti_core.llm_client import OpenAIClient, LLMConfig

config = LLMConfig(
    api_key="sk-or-v1-...",
    model="meta-llama/llama-3-8b-instruct",
    base_url="https://openrouter.ai/api/v1"
)

llm_client = OpenAIClient(config=config)
graphiti = Graphiti(uri, user, password, llm_client=llm_client)

# This will fail because OpenRouter doesn't have /v1/responses endpoint
# Error: 404 Not Found - https://openrouter.ai/api/v1/responses
```

## Current Workaround (Documented)

The README documents using `OpenAIGenericClient` with Ollama:

```python
from graphiti_core.llm_client.openai_generic_client import OpenAIGenericClient
from graphiti_core.llm_client.config import LLMConfig

llm_config = LLMConfig(
    api_key="ollama",
    model="deepseek-r1:7b",
    base_url="http://localhost:11434/v1"
)

llm_client = OpenAIGenericClient(config=llm_config)
```

### Limitations of Current Workaround

- `OpenAIGenericClient` **doesn't support structured outputs with Pydantic models**
- It only returns raw JSON and manually validates schemas
- It's not the recommended/default client
- It's **not exported** in the public API (`graphiti_core.llm_client`)
- Users must know to import from the internal module path

## Recommended Solutions

### Priority 1: Quick Wins (High Priority)

#### 1.1 Export `OpenAIGenericClient` in Public API

**File**: `graphiti_core/llm_client/__init__.py`

**Current**:
```python
from .client import LLMClient
from .config import LLMConfig
from .errors import RateLimitError
from .openai_client import OpenAIClient

__all__ = ['LLMClient', 'OpenAIClient', 'LLMConfig', 'RateLimitError']
```

**Proposed**:
```python
from .client import LLMClient
from .config import LLMConfig
from .errors import RateLimitError
from .openai_client import OpenAIClient
from .openai_generic_client import OpenAIGenericClient

__all__ = ['LLMClient', 'OpenAIClient', 'OpenAIGenericClient', 'LLMConfig', 'RateLimitError']
```

#### 1.2 Add Clear Documentation

**File**: `README.md`

Add a dedicated section:

```markdown
### Using OpenAI-Compatible Endpoints (OpenRouter, NagaAI, Together.ai, etc.)

Most OpenAI-compatible services only support the standard Chat Completions API,
not OpenAI's newer Responses API. Use `OpenAIGenericClient` for these services:

**OpenRouter Example**:
```python
from graphiti_core import Graphiti
from graphiti_core.llm_client import OpenAIGenericClient, LLMConfig

config = LLMConfig(
    api_key="sk-or-v1-...",
    model="meta-llama/llama-3-8b-instruct",
    base_url="https://openrouter.ai/api/v1"
)

llm_client = OpenAIGenericClient(config=config)
graphiti = Graphiti(uri, user, password, llm_client=llm_client)
```

**Together.ai Example**:
```python
config = LLMConfig(
    api_key="your-together-api-key",
    model="meta-llama/Llama-3-70b-chat-hf",
    base_url="https://api.together.xyz/v1"
)
llm_client = OpenAIGenericClient(config=config)
```

**Note**: `OpenAIGenericClient` has limited structured output support compared to
the default `OpenAIClient`. It uses JSON mode instead of Pydantic schema validation.
```

#### 1.3 Add Better Error Messages

**File**: `graphiti_core/llm_client/openai_client.py`

Add error handling that detects the issue:

```python
async def _create_structured_completion(self, ...):
    try:
        response = await self.client.responses.parse(...)
        return response
    except openai.NotFoundError as e:
        if self.config.base_url and "api.openai.com" not in self.config.base_url:
            raise Exception(
                f"The OpenAI Responses API (/v1/responses) is not available at {self.config.base_url}. "
                f"Most OpenAI-compatible services only support /v1/chat/completions. "
                f"Please use OpenAIGenericClient instead of OpenAIClient for custom endpoints. "
                f"See: https://help.getzep.com/graphiti/guides/custom-endpoints"
            ) from e
        raise
```

### Priority 2: Better UX (Medium Priority)

#### 2.1 Add Auto-Detection Logic

**File**: `graphiti_core/llm_client/config.py`

```python
class LLMConfig:
    def __init__(
        self,
        api_key: str | None = None,
        model: str | None = None,
        base_url: str | None = None,
        temperature: float = DEFAULT_TEMPERATURE,
        max_tokens: int = DEFAULT_MAX_TOKENS,
        small_model: str | None = None,
        use_responses_api: bool | None = None,  # NEW: Auto-detect if None
    ):
        self.base_url = base_url
        self.api_key = api_key
        self.model = model
        self.small_model = small_model
        self.temperature = temperature
        self.max_tokens = max_tokens
        
        # Auto-detect API style based on base_url
        if use_responses_api is None:
            self.use_responses_api = self._should_use_responses_api()
        else:
            self.use_responses_api = use_responses_api
    
    def _should_use_responses_api(self) -> bool:
        """Determine if we should use the Responses API based on base_url."""
        if self.base_url is None:
            return True  # Default OpenAI
        
        # Known services that support Responses API
        supported_services = ["api.openai.com", "azure.com"]
        return any(service in self.base_url for service in supported_services)
```

#### 2.2 Create a Unified Smart Client

**Option A**: Modify `OpenAIClient` to Fall Back

```python
class OpenAIClient(BaseOpenAIClient):
    def __init__(self, config: LLMConfig | None = None, ...):
        super().__init__(config, ...)
        if config is None:
            config = LLMConfig()
        
        self.use_responses_api = config.use_responses_api
        self.client = AsyncOpenAI(api_key=config.api_key, base_url=config.base_url)
    
    async def _create_structured_completion(self, ...):
        if self.use_responses_api:
            # Use responses.parse() for OpenAI native
            return await self.client.responses.parse(...)
        else:
            # Fall back to chat.completions with JSON schema for compatibility
            return await self.client.chat.completions.create(
                model=model,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
                response_format={
                    "type": "json_schema",
                    "json_schema": {
                        "name": response_model.__name__,
                        "schema": response_model.model_json_schema(),
                        "strict": False
                    }
                }
            )
```

**Option B**: Create a Factory Function

```python
# graphiti_core/llm_client/__init__.py

def create_openai_client(
    config: LLMConfig | None = None,
    cache: bool = False,
    **kwargs
) -> LLMClient:
    """
    Factory to create the appropriate OpenAI-compatible client.
    
    Automatically selects between OpenAIClient (for native OpenAI)
    and OpenAIGenericClient (for OpenAI-compatible services).
    
    Args:
        config: LLM configuration including base_url
        cache: Whether to enable caching
        **kwargs: Additional arguments passed to the client
    
    Returns:
        LLMClient: Either OpenAIClient or OpenAIGenericClient
    
    Example:
        >>> # Automatically uses OpenAIGenericClient for OpenRouter
        >>> config = LLMConfig(
        ...     api_key="sk-or-v1-...",
        ...     model="meta-llama/llama-3-8b-instruct",
        ...     base_url="https://openrouter.ai/api/v1"
        ... )
        >>> client = create_openai_client(config)
    """
    if config is None:
        config = LLMConfig()
    
    # Auto-detect based on base_url
    if config.base_url is None or "api.openai.com" in config.base_url:
        return OpenAIClient(config, cache, **kwargs)
    else:
        return OpenAIGenericClient(config, cache, **kwargs)
```

#### 2.3 Enhance `OpenAIGenericClient` with Better Structured Output Support

**File**: `graphiti_core/llm_client/openai_generic_client.py`

```python
async def _generate_response(
    self,
    messages: list[Message],
    response_model: type[BaseModel] | None = None,
    max_tokens: int = DEFAULT_MAX_TOKENS,
    model_size: ModelSize = ModelSize.medium,
) -> dict[str, typing.Any]:
    openai_messages: list[ChatCompletionMessageParam] = []
    for m in messages:
        m.content = self._clean_input(m.content)
        if m.role == 'user':
            openai_messages.append({'role': 'user', 'content': m.content})
        elif m.role == 'system':
            openai_messages.append({'role': 'system', 'content': m.content})

    try:
        # Try to use json_schema format (supported by more providers)
        if response_model:
            response = await self.client.chat.completions.create(
                model=self.model or DEFAULT_MODEL,
                messages=openai_messages,
                temperature=self.temperature,
                max_tokens=max_tokens or self.max_tokens,
                response_format={
                    "type": "json_schema",
                    "json_schema": {
                        "name": response_model.__name__,
                        "schema": response_model.model_json_schema(),
                        "strict": False  # Most providers don't support strict mode
                    }
                }
            )
        else:
            response = await self.client.chat.completions.create(
                model=self.model or DEFAULT_MODEL,
                messages=openai_messages,
                temperature=self.temperature,
                max_tokens=max_tokens or self.max_tokens,
                response_format={'type': 'json_object'},
            )
        
        result = response.choices[0].message.content or '{}'
        return json.loads(result)
    except Exception as e:
        logger.error(f'Error in generating LLM response: {e}')
        raise
```

### Priority 3: Nice to Have (Low Priority)

#### 3.1 Provider-Specific Clients

Create convenience clients for popular providers:

```python
# graphiti_core/llm_client/openrouter_client.py
class OpenRouterClient(OpenAIGenericClient):
    """Pre-configured client for OpenRouter.
    
    Example:
        >>> client = OpenRouterClient(
        ...     api_key="sk-or-v1-...",
        ...     model="meta-llama/llama-3-8b-instruct"
        ... )
    """
    def __init__(
        self,
        api_key: str,
        model: str,
        temperature: float = DEFAULT_TEMPERATURE,
        max_tokens: int = DEFAULT_MAX_TOKENS,
        **kwargs
    ):
        config = LLMConfig(
            api_key=api_key,
            model=model,
            base_url="https://openrouter.ai/api/v1",
            temperature=temperature,
            max_tokens=max_tokens
        )
        super().__init__(config=config, **kwargs)
```

```python
# graphiti_core/llm_client/together_client.py
class TogetherClient(OpenAIGenericClient):
    """Pre-configured client for Together.ai.
    
    Example:
        >>> client = TogetherClient(
        ...     api_key="your-together-key",
        ...     model="meta-llama/Llama-3-70b-chat-hf"
        ... )
    """
    def __init__(
        self,
        api_key: str,
        model: str,
        temperature: float = DEFAULT_TEMPERATURE,
        max_tokens: int = DEFAULT_MAX_TOKENS,
        **kwargs
    ):
        config = LLMConfig(
            api_key=api_key,
            model=model,
            base_url="https://api.together.xyz/v1",
            temperature=temperature,
            max_tokens=max_tokens
        )
        super().__init__(config=config, **kwargs)
```

#### 3.2 Provider Compatibility Matrix

Add to documentation:

| Provider | Standard Client | Generic Client | Structured Outputs | Notes |
|----------|----------------|----------------|-------------------|-------|
| OpenAI | ✅ `OpenAIClient` | ✅ | ✅ Full (Responses API) | Recommended: Use `OpenAIClient` |
| Azure OpenAI | ✅ `AzureOpenAILLMClient` | ✅ | ✅ Full (Responses API) | Requires API version 2024-08-01-preview+ |
| OpenRouter | ❌ | ✅ `OpenAIGenericClient` | ⚠️ Limited (JSON Schema) | Use `OpenAIGenericClient` |
| Together.ai | ❌ | ✅ `OpenAIGenericClient` | ⚠️ Limited (JSON Schema) | Use `OpenAIGenericClient` |
| Ollama | ❌ | ✅ `OpenAIGenericClient` | ⚠️ Limited (JSON mode) | Local deployment |
| Groq | ❌ | ✅ `OpenAIGenericClient` | ⚠️ Limited (JSON Schema) | Very fast inference |
| Perplexity | ❌ | ✅ `OpenAIGenericClient` | ⚠️ Limited (JSON mode) | Primarily for search |

## Testing Recommendations

### Unit Tests

1. **Endpoint detection logic**
   ```python
   def test_should_use_responses_api():
       # OpenAI native should use Responses API
       config = LLMConfig(base_url="https://api.openai.com/v1")
       assert config.use_responses_api is True
       
       # Custom endpoints should not
       config = LLMConfig(base_url="https://openrouter.ai/api/v1")
       assert config.use_responses_api is False
   ```

2. **Client selection**
   ```python
   def test_create_openai_client_auto_selection():
       # Should return OpenAIClient for OpenAI
       config = LLMConfig(api_key="test")
       client = create_openai_client(config)
       assert isinstance(client, OpenAIClient)
       
       # Should return OpenAIGenericClient for others
       config = LLMConfig(api_key="test", base_url="https://openrouter.ai/api/v1")
       client = create_openai_client(config)
       assert isinstance(client, OpenAIGenericClient)
   ```

### Integration Tests

1. **Mock server tests** with responses for both endpoints
2. **Real provider tests** (optional, may require API keys):
   - OpenRouter
   - Together.ai
   - Ollama (local)

### Manual Testing Checklist

- [ ] OpenRouter with Llama models
- [ ] Together.ai with various models
- [ ] Ollama with local models
- [ ] Groq with fast models
- [ ] Verify error messages are helpful
- [ ] Test both structured and unstructured outputs

## Summary of Issues

| Issue | Current State | Impact | Priority |
|-------|---------------|--------|----------|
| `/v1/responses` endpoint usage | Used by default `OpenAIClient` | **BREAKS** all non-OpenAI providers | High |
| `OpenAIGenericClient` not exported | Hidden from public API | Users can't easily use it | High |
| Poor error messages | Generic 404 errors | Confusing for users | High |
| No auto-detection | Must manually choose client | Poor DX | Medium |
| Limited docs | Only Ollama example | Users don't know how to configure other providers | High |
| No structured output in Generic client | Only supports loose JSON | Reduced type safety for custom endpoints | Medium |
| No provider-specific helpers | Generic configuration only | More setup required | Low |

## Implementation Roadmap

### Phase 1: Quick Fixes (1-2 days)
1. Export `OpenAIGenericClient` in public API
2. Add documentation section for custom endpoints
3. Improve error messages in `OpenAIClient`
4. Add examples for OpenRouter, Together.ai

### Phase 2: Enhanced Support (3-5 days)
1. Add auto-detection logic to `LLMConfig`
2. Create factory function for client selection
3. Enhance `OpenAIGenericClient` with better JSON schema support
4. Add comprehensive tests

### Phase 3: Polish (2-3 days)
1. Create provider-specific client classes
2. Build compatibility matrix documentation
3. Add integration tests with real providers
4. Update all examples and guides

## References

- OpenAI SDK v1.91.0+ Responses API: https://platform.openai.com/docs/api-reference/responses
- OpenAI Chat Completions API: https://platform.openai.com/docs/api-reference/chat
- OpenRouter API: https://openrouter.ai/docs
- Together.ai API: https://docs.together.ai/docs/openai-api-compatibility
- Ollama OpenAI compatibility: https://github.com/ollama/ollama/blob/main/docs/openai.md

## Contributing

If you're implementing these changes, please ensure:

1. All changes follow the repository guidelines in `AGENTS.md`
2. Run `make format` before committing
3. Run `make lint` and `make test` to verify changes
4. Update documentation for any new public APIs
5. Add examples demonstrating the new functionality

## Questions or Issues?

- Open an issue: https://github.com/getzep/graphiti/issues
- Discussion: https://github.com/getzep/graphiti/discussions
- Documentation: https://help.getzep.com/graphiti
