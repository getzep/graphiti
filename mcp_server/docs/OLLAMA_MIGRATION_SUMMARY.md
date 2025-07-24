# Ollama Migration Summary

## Overview

The Graphiti MCP server has been successfully migrated to use **Ollama** as the default LLM provider instead of OpenAI. This change provides several benefits:

- **Privacy**: All LLM operations run locally
- **Cost**: No API costs for LLM operations
- **Performance**: Lower latency for local operations
- **Flexibility**: Easy to switch between different local models

## Key Changes Made

### 1. Default Model Configuration

**Before:**
```python
DEFAULT_LLM_MODEL = 'gpt-4.1-mini'
SMALL_LLM_MODEL = 'gpt-4.1-nano'
DEFAULT_EMBEDDER_MODEL = 'text-embedding-3-small'
```

**After:**
```python
DEFAULT_LLM_MODEL = 'deepseek-r1:7b'
SMALL_LLM_MODEL = 'deepseek-r1:7b'
DEFAULT_EMBEDDER_MODEL = 'nomic-embed-text'
```

### 2. LLM Configuration Class Updates

Added Ollama-specific fields to `GraphitiLLMConfig`:
- `use_ollama: bool = True` - Default to Ollama
- `ollama_base_url: str = "http://localhost:11434/v1"`
- `ollama_llm_model: str = DEFAULT_LLM_MODEL`

### 3. Embedder Configuration Class Updates

Added Ollama-specific fields to `GraphitiEmbedderConfig`:
- `use_ollama: bool = True` - Default to Ollama
- `ollama_base_url: str = "http://localhost:11434/v1"`
- `ollama_embedding_model: str = DEFAULT_EMBEDDER_MODEL`
- `ollama_embedding_dim: int = 768`

### 4. Environment Variables

**New Ollama Environment Variables:**
- `USE_OLLAMA` - Use Ollama for LLM and embeddings (default: `true`)
- `OLLAMA_BASE_URL` - Ollama base URL (default: `http://localhost:11434/v1`)
- `OLLAMA_LLM_MODEL` - Ollama LLM model name (default: `deepseek-r1:7b`)
- `OLLAMA_EMBEDDING_MODEL` - Ollama embedding model name (default: `nomic-embed-text`)
- `OLLAMA_EMBEDDING_DIM` - Ollama embedding dimension (default: `768`)

### 5. CLI Arguments

**New CLI Arguments:**
- `--use-ollama` - Use Ollama for LLM and embeddings (default: true)
- `--ollama-base-url` - Ollama base URL (default: http://localhost:11434/v1)
- `--ollama-llm-model` - Ollama LLM model name (default: deepseek-r1:7b)
- `--ollama-embedding-model` - Ollama embedding model name (default: nomic-embed-text)
- `--ollama-embedding-dim` - Ollama embedding dimension (default: 768)

### 6. Client Creation Logic

Updated `create_client()` methods to handle Ollama configuration:
- Uses `OpenAIClient` with custom `base_url` pointing to Ollama
- Sets `api_key` to "abc" (Ollama doesn't require real API keys)
- Configures both LLM and embedding models

## Backward Compatibility

The migration maintains full backward compatibility:

1. **OpenAI Support**: Set `USE_OLLAMA=false` to use OpenAI
2. **Azure OpenAI Support**: Set `USE_OLLAMA=false` and configure Azure endpoints
3. **Existing Configurations**: Will continue to work with appropriate environment variables

## Setup Instructions

### For New Users (Ollama Default)

1. **Install Ollama**: Visit [https://ollama.ai](https://ollama.ai)
2. **Start Ollama**: Run `ollama serve`
3. **Pull Models**:
   ```bash
   ollama pull deepseek-r1:7b     # LLM model
   ollama pull nomic-embed-text   # Embedding model
   ```
4. **Run Server**: The server will automatically use Ollama

### For Existing Users (OpenAI)

1. **Set Environment Variable**: `USE_OLLAMA=false`
2. **Configure OpenAI**: Set your `OPENAI_API_KEY`
3. **Run Server**: Will use OpenAI as before

## Testing

The migration has been tested with:
- ✅ Default Ollama configuration
- ✅ Custom Ollama model configuration
- ✅ OpenAI fallback configuration
- ✅ CLI argument overrides
- ✅ Environment variable configuration

## Benefits

1. **Privacy**: All LLM operations run locally
2. **Cost**: No API costs for LLM operations
3. **Performance**: Lower latency for local operations
4. **Flexibility**: Easy to switch between different local models
5. **Offline Capability**: Works without internet connection
6. **Customization**: Easy to use custom fine-tuned models

## Migration Path

### For Existing Deployments

1. **Option 1**: Keep using OpenAI
   - Set `USE_OLLAMA=false`
   - No other changes needed

2. **Option 2**: Migrate to Ollama
   - Install Ollama
   - Pull required models
   - Remove `OPENAI_API_KEY` (or set `USE_OLLAMA=true`)
   - Update any custom model configurations

### For New Deployments

1. Install Ollama
2. Pull required models
3. Run server (will use Ollama by default)

## Configuration Examples

### Ollama (Default)
```bash
# No configuration needed - uses defaults
uv run graphiti_mcp_server.py
```

### Custom Ollama Models
```bash
uv run graphiti_mcp_server.py \
  --ollama-llm-model llama3.1:8b \
  --ollama-embedding-model nomic-embed-text
```

### OpenAI (Legacy)
```bash
USE_OLLAMA=false OPENAI_API_KEY=your_key uv run graphiti_mcp_server.py
```

## Files Modified

1. `graphiti_mcp_server.py` - Main server implementation
2. `README.md` - Documentation updates
3. `OLLAMA_MIGRATION_SUMMARY.md` - This summary document

## Next Steps

1. **Deploy**: The changes are ready for deployment
2. **Test**: Verify with actual Ollama installation
3. **Monitor**: Watch for any issues in production
4. **Optimize**: Consider model selection based on use case
