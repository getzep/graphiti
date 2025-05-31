"""
Configuration for AI services used in fact extraction.
"""
import os
from dataclasses import dataclass
from typing import Optional
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Load settings and API key
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

@dataclass
class ExtractionConfig:
    """Configuration for a specific type of extraction (facts, emotions, entities)"""
    model: str
    temperature: float
    max_tokens: Optional[int] = None

# Configuration for different types of extractions
FACTS_CONFIG = ExtractionConfig(
    model="gpt-4o-mini",
    temperature=0  # Lower temperature for factual accuracy
)

EMOTIONS_CONFIG = ExtractionConfig(
    model="gpt-4o-mini", 
    temperature=0.4  # Higher temperature for emotional nuance
)

ENTITIES_CONFIG = ExtractionConfig(
    model="gpt-4o-mini",
    temperature=0.2  # Medium temperature for entity recognition
)

# Backward compatibility - default values
OPENAI_MODEL = FACTS_CONFIG.model
TEMPERATURE = FACTS_CONFIG.temperature
