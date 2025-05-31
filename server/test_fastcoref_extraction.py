"""
Test script for the new FastCoref-based extraction system.
This demonstrates how the new modular system works.
"""
import asyncio
import sys
import os

# Add the server directory to the path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from graph_service.routers.ai.extraction import extract_facts_emotions_entities
from graph_service.routers.ai.coreference_resolver import get_coreference_resolver

async def test_basic_extraction():
    """Test basic extraction functionality."""
    print("=== Test 1: Basic Extraction ===")
    
    message = "Poszedłem wczoraj z Anną do kina. Ona bardzo lubiła film."
    existing_entities = ["Anna"]
    existing_emotions = ["radość", "zadowolenie"]
    
    result = await extract_facts_emotions_entities(
        message_content=message,
        existing_entities=existing_entities,
        existing_emotions=existing_emotions,
        chat_history=None
    )
    
    print(f"Original text: {message}")
    print(f"Resolved text: {result['resolved_text']}")
    print(f"Facts: {result['facts']}")
    print(f"Emotions: {result['emotions']}")
    print(f"Entities: {result['entities']}")
    print(f"Coreference info: {result['coreference_info']}")
    print()


async def test_with_chat_history():
    """Test extraction with chat history for context."""
    print("=== Test 2: Extraction with Chat History ===")
    
    chat_history = [
        {"role": "user", "content": "Spotkałem się wczoraj z Jankiem."},
        {"role": "assistant", "content": "To miło! Jak się miewa Janek?"},
        {"role": "user", "content": "Dobrze, rozmawialiśmy o pracy."}
    ]
    
    message = "On opowiadał mi o swoim nowym projekcie. Był bardzo podekscytowany."
    existing_entities = ["Janek"]
    existing_emotions = ["podekscytowanie"]
    
    result = await extract_facts_emotions_entities(
        message_content=message,
        existing_entities=existing_entities,
        existing_emotions=existing_emotions,
        chat_history=chat_history
    )
    
    print(f"Chat history: {[msg['content'] for msg in chat_history]}")
    print(f"Current message: {message}")
    print(f"Resolved text: {result['resolved_text']}")
    print(f"Facts: {result['facts']}")
    print(f"Emotions: {result['emotions']}")
    print(f"Entities: {result['entities']}")
    print()


async def test_english_example():
    """Test with English text."""
    print("=== Test 3: English Text ===")
    
    chat_history = [
        {"role": "user", "content": "I met Sarah at the coffee shop yesterday."},
        {"role": "assistant", "content": "How is Sarah doing?"}
    ]
    
    message = "She told me about her new job. She seemed really excited about it."
    existing_entities = ["Sarah", "coffee shop"]
    existing_emotions = ["excitement", "happiness"]
    
    result = await extract_facts_emotions_entities(
        message_content=message,
        existing_entities=existing_entities,
        existing_emotions=existing_emotions,
        chat_history=chat_history
    )
    
    print(f"Chat history: {[msg['content'] for msg in chat_history]}")
    print(f"Current message: {message}")
    print(f"Resolved text: {result['resolved_text']}")
    print(f"Facts: {result['facts']}")
    print(f"Emotions: {result['emotions']}")
    print(f"Entities: {result['entities']}")
    print()


async def test_coreference_resolver_directly():
    """Test the coreference resolver directly."""
    print("=== Test 4: Direct Coreference Resolver Test ===")
    
    resolver = get_coreference_resolver()
    print(f"Coreference resolver available: {resolver.is_available()}")
    
    if resolver.is_available():
        context = ["Spotkałem wczoraj Martę w parku."]
        text = "Ona była bardzo szczęśliwa."
        
        result = resolver.resolve_coreferences_and_extract_entities(
            text=text,
            context_history=context,
            existing_entities=["Marta"]
        )
        
        print(f"Context: {context}")
        print(f"Original: {text}")
        print(f"Resolved: {result['resolved_text']}")
        print(f"Entities: {result['entities']}")
        print(f"Clusters: {result['coreference_clusters']}")
    else:
        print("FastCoref not available - install required models")
    print()


async def main():
    """Run all tests."""
    print("Testing FastCoref-based extraction system...")
    print("=" * 50)
    
    try:
        await test_basic_extraction()
        await test_with_chat_history() 
        await test_english_example()
        await test_coreference_resolver_directly()
        
        print("All tests completed!")
        
    except Exception as e:
        print(f"Error during testing: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    # Note: This requires OpenAI API key to be configured
    print("Note: Make sure your OpenAI API key is configured in the config file.")
    print("Also ensure FastCoref dependencies are installed.")
    print()
    
    asyncio.run(main())
