# FastCoref Integration for Entity Extraction

This implementation integrates FastCoref for advanced coreference resolution and entity extraction, replacing the OpenAI-based entity extraction while keeping OpenAI for facts and emotions.

## Overview

The new system consists of two main components:

1. **FastCoref-based Entity Extraction** (`coreference_resolver.py`)
   - Resolves pronouns to actual entities (e.g., "on" → "Janek")
   - Extracts entities from text using both FastCoref and spaCy
   - Provides resolved text with coreferences replaced

2. **Modular Extraction System** (`extraction.py`)
   - Uses FastCoref for entities and coreference resolution
   - Uses OpenAI for facts and emotions extraction
   - Cleaner, more maintainable code structure

## Key Features

### ✅ Coreference Resolution
- Resolves pronouns like "on", "ona", "he", "she" to actual entity names
- Uses full conversation context for better accuracy
- Returns both original and resolved text

### ✅ Entity Extraction
- FastCoref-based entity detection with spaCy fallback
- Merges with existing known entities
- Filters out pronouns and common words

### ✅ Modular Design
- Separate functions for different extraction tasks
- Easy to test and maintain
- Backward compatibility maintained

### ✅ Multi-language Support
- Polish and English coreference resolution
- Automatic model selection based on availability

## API Changes

### New Main Function
```python
async def extract_facts_emotions_entities(
    message_content: str, 
    existing_emotions: List[str] = None, 
    existing_entities: List[str] = None,
    chat_history: Union[List[Dict], str] = None
) -> Dict[str, Any]:
```

### Enhanced Return Format
```python
{
    "facts": ["User poszedł z Jankiem do baru"],
    "emotions": ["zadowolenie"],
    "entities": ["Janek", "bar"],
    "resolved_text": "User poszedł z Jankiem do baru",  # NEW
    "usage": {...},
    "coreference_info": {  # NEW
        "original_text": "Poszedłem z nim do baru",
        "clusters": [...]
    }
}
```

## Installation Requirements

### 1. Install FastCoref
```bash
pip install fastcoref
```

### 2. Download spaCy Models
```bash
# For Polish (recommended)
python -m spacy download pl_core_news_sm

# For English (fallback)
python -m spacy download en_core_web_sm
```

### 3. GPU Support (Optional)
For better performance, install CUDA-compatible PyTorch:
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

## Usage Examples

### Basic Usage
```python
from graph_service.routers.ai.extraction import extract_facts_emotions_entities

result = await extract_facts_emotions_entities(
    message_content="Poszedłem z nim do kina",
    existing_entities=["Janek"],
    chat_history=[
        {"role": "user", "content": "Spotkałem wczoraj Janka"}
    ]
)

print(f"Resolved: {result['resolved_text']}")
# Output: "Poszedłem z Jankiem do kina"
```

### Direct Coreference Resolution
```python
from graph_service.routers.ai.coreference_resolver import get_coreference_resolver

resolver = get_coreference_resolver()
result = resolver.resolve_coreferences_and_extract_entities(
    text="Ona była bardzo szczęśliwa",
    context_history=["Spotkałem wczoraj Anię"],
    existing_entities=["Ania"]
)
```

## Testing

Run the test script to verify functionality:

```bash
cd server
python test_fastcoref_extraction.py
```

## Performance Considerations

### Memory Usage
- FastCoref models are loaded once and reused (singleton pattern)
- Models run on CPU by default for stability
- Memory usage: ~500MB-1GB depending on model

### Speed
- Initial model loading: 5-10 seconds
- Per-request processing: 100-500ms
- Caching improves subsequent requests

### Fallback Strategy
1. Try FastCoref for coreference resolution
2. Fall back to spaCy for entity extraction
3. Graceful degradation if models unavailable

## Configuration

### Model Selection
Modify `coreference_resolver.py` to change models:
```python
# Default model
model_name = "biu-nlp/f-coref"

# Alternative models
# model_name = "biu-nlp/f-coref-large"  # Better accuracy, slower
```

### Device Selection
```python
# CPU (default, more stable)
self.model = FCoref(device='cpu')

# GPU (faster, requires CUDA)
self.model = FCoref(device='cuda')
```

## Migration Guide

### From Old System
The old function is still available as `extract_facts_emotions_entities_legacy()` for backward compatibility.

### Changes Required
1. Update imports if using internal functions
2. Handle new return fields (`resolved_text`, `coreference_info`)
3. Update tests to expect new format

### Database Schema
Consider updating Neo4j schema to store:
- `resolved_text` alongside original text
- Coreference cluster information
- Entity confidence scores

## Troubleshooting

### Common Issues

1. **FastCoref not loading**
   - Check CUDA availability if using GPU
   - Ensure sufficient memory (>2GB RAM)
   - Try CPU mode: `FCoref(device='cpu')`

2. **spaCy model missing**
   - Download required models: `python -m spacy download en_core_web_sm`
   - Check model compatibility with spaCy version

3. **Poor coreference resolution**
   - Ensure sufficient context in chat_history
   - Verify entity names in existing_entities
   - Consider using larger FastCoref model

### Debug Mode
Enable detailed logging:
```python
import logging
logging.basicConfig(level=logging.DEBUG)
```

## Future Enhancements

### Planned Features
- [ ] Custom entity type classification
- [ ] Confidence scores for resolved entities
- [ ] Multi-document coreference resolution
- [ ] Integration with knowledge graphs

### Performance Optimizations
- [ ] Model quantization for faster inference
- [ ] Batch processing for multiple messages
- [ ] Caching layer for common patterns

## Contributing

When modifying the coreference resolution system:

1. Update tests in `test_fastcoref_extraction.py`
2. Maintain backward compatibility
3. Document any new configuration options
4. Test with both Polish and English text
