#!/usr/bin/env python3
"""
Test script to check the response structure from fact extraction.
"""
import asyncio
import json
from uuid import uuid4
from datetime import datetime

# Mock classes to simulate the structure
class MockMessage:
    def __init__(self):
        self.uuid = str(uuid4())
        self.content = "I love programming with Python and AI. It makes me feel excited and motivated."
        self.role = "user"
        self.timestamp = datetime.now()

class MockUsage:
    def __init__(self):
        self.input_tokens = 150
        self.output_tokens = 75
        self.total_tokens = 225
        self.model = "gpt-4.1-mini"
        self.temperature = 0.25

async def test_response_structure():
    """Test what the current response structure looks like."""
    
    # Simulate the current response from factExtractor.py
    mock_message = MockMessage()
    mock_usage = MockUsage()
    
    # This simulates what get_current_message_data would return after our changes
    current_message_data = {
        "message_facts": [
            "Programming with Python is enjoyable",
            "AI technology is advancing rapidly",
            "Python is great for AI development"
        ],
        "message_emotions": [
            "excited",
            "motivated", 
            "enthusiastic"
        ],
        "message_entities": [
            {
                "text": "Python",
                "related_emotions": ["excited", "motivated"],
                "related_facts": ["Python is great for AI development", "Programming with Python is enjoyable"]
            },
            {
                "text": "AI",
                "related_emotions": ["excited", "enthusiastic"],
                "related_facts": ["AI technology is advancing rapidly", "Python is great for AI development"]
            },
            {
                "text": "programming",
                "related_emotions": ["motivated"],
                "related_facts": ["Programming with Python is enjoyable"]
            }
        ]
    }
      # This is what factExtractor.py returns
    response = {
        "facts": current_message_data["message_facts"],
        "emotions": current_message_data["message_emotions"], 
        "entities": current_message_data["message_entities"],
        "tokens": {
            "input_tokens": mock_usage.input_tokens,
            "output_tokens": mock_usage.output_tokens, 
            "total_tokens": mock_usage.total_tokens,
            "model": mock_usage.model,
            "temperature": mock_usage.temperature
        },
        "message_uuid": mock_message.uuid
    }
      print("=== CURRENT RESPONSE STRUCTURE ===")
    print(json.dumps(response, indent=2, ensure_ascii=False))
    print("\n=== STRUCTURE BREAKDOWN ===")
    print(f"Facts: {len(response['facts'])} items (simple strings)")
    print(f"Emotions: {len(response['emotions'])} items (simple strings)")
    print(f"Entities: {len(response['entities'])} items (objects with text + related arrays)")
    print(f"Tokens: Contains usage info (input_tokens, output_tokens, total_tokens, model, temperature)")
    print(f"Message UUID: {response['message_uuid']}")
    print(f"Model: {response['tokens']['model']}")
    print(f"Temperature: {response['tokens']['temperature']}")
    
    print("\n=== ENTITY STRUCTURE DETAILS ===")
    for i, entity in enumerate(response['entities']):
        print(f"Entity {i+1}:")
        print(f"  - text: '{entity['text']}'")
        print(f"  - related_emotions: {entity['related_emotions']} ({len(entity['related_emotions'])} items)")
        print(f"  - related_facts: {entity['related_facts']} ({len(entity['related_facts'])} items)")

if __name__ == "__main__":
    asyncio.run(test_response_structure())
