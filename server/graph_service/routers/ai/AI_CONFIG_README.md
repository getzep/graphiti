# AI Configuration Guide

## Overview
This module now supports separate configurations for different types of extractions: facts, emotions, and entities. Each type can use its own model and temperature settings for optimal performance.

## Environment Variables

You can configure each extraction type separately using environment variables:

### Facts Extraction
- `FACTS_MODEL`: Model to use for fact extraction (default: "gpt-4.1-mini")
- `FACTS_TEMPERATURE`: Temperature for fact extraction (default: 0.1)

### Emotions Extraction
- `EMOTIONS_MODEL`: Model to use for emotion extraction (default: "gpt-4.1-mini") 
- `EMOTIONS_TEMPERATURE`: Temperature for emotion extraction (default: 0.4)

### Entities/People Extraction
- `ENTITIES_MODEL`: Model to use for entity extraction (default: "gpt-4.1-mini")
- `ENTITIES_TEMPERATURE`: Temperature for entity extraction (default: 0.2)

### Global Configuration
- `OPENAI_API_KEY`: Your OpenAI API key (required)

## Temperature Guidelines

### Facts (default: 0.1)
- **Low temperature** ensures consistency and accuracy
- Facts should be objective and reproducible
- Minimal creativity needed

### Emotions (default: 0.4)
- **Higher temperature** allows for nuanced emotion detection
- Emotions can be subjective and context-dependent
- Some creativity helps capture subtle emotional undertones

### Entities/People (default: 0.2)
- **Medium temperature** balances accuracy with flexibility
- Entity recognition benefits from some variation
- Helps with name variations and informal references

## Example Configuration

```bash
# Environment variables
export OPENAI_API_KEY="your-api-key-here"

# Facts: Use precise, consistent model
export FACTS_MODEL="gpt-4.1-mini"
export FACTS_TEMPERATURE="0.05"

# Emotions: Use creative model for nuance
export EMOTIONS_MODEL="gpt-4.1"
export EMOTIONS_TEMPERATURE="0.5"

# Entities: Use balanced approach
export ENTITIES_MODEL="gpt-4.1-mini"
export ENTITIES_TEMPERATURE="0.2"
```

## Usage in Code

The configuration is automatically applied when calling the extraction functions:

```python
from .config import FACTS_CONFIG, EMOTIONS_CONFIG, ENTITIES_CONFIG

# Each config contains:
# - model: str
# - temperature: float
# - max_tokens: Optional[int]

# Example:
print(f"Facts model: {FACTS_CONFIG.model}, temp: {FACTS_CONFIG.temperature}")
print(f"Emotions model: {EMOTIONS_CONFIG.model}, temp: {EMOTIONS_CONFIG.temperature}")
print(f"Entities model: {ENTITIES_CONFIG.model}, temp: {ENTITIES_CONFIG.temperature}")
```

## Response Format

The extraction results now include detailed information about which models and temperatures were used:

```python
{
    "facts": ["fact1", "fact2"],
    "emotions": ["emotion1", "emotion2"], 
    "entities": ["person1", "person2"],
    "input_tokens": 150,
    "output_tokens": 75,
    "total_tokens": 225,
    "models": {
        "facts": "gpt-4.1-mini",
        "emotions": "gpt-4.1-mini", 
        "entities": "gpt-4.1-mini"
    },
    "temperatures": {
        "facts": 0.1,
        "emotions": 0.4,
        "entities": 0.2
    }
}
```

## Migration from Old Config

The old configuration variables are still supported for backward compatibility:
- `OPENAI_MODEL` maps to `FACTS_CONFIG.model`
- `TEMPERATURE` maps to `FACTS_CONFIG.temperature`

However, it's recommended to migrate to the new granular configuration for better results.
