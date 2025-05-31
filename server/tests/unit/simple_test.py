"""
Simple test to verify FastCoref integration works.
"""
import asyncio
import sys
import os

# Add the server directory to the path
sys.path.append(os.path.dirname(__file__))

async def test_simple_case():
    """Test with a simple case that doesn't require OpenAI."""
    print("=== Simple Test without OpenAI ===")
    
    # Test only the coreference resolver part
    from graph_service.routers.ai.coreference_resolver import get_coreference_resolver
    
    resolver = get_coreference_resolver()
    print(f"FastCoref available: {resolver.is_available()}")
    
    if resolver.is_available():
        # Test English
        print("\n--- English Test ---")
        context = ["I met Sarah yesterday at the coffee shop."]
        text = "She was very happy about her new job."
        
        result = resolver.resolve_coreferences_and_extract_entities(
            text=text,
            context_history=context,
            existing_entities=["Sarah", "coffee shop"]
        )
        
        print(f"Context: {context}")
        print(f"Original: {text}")
        print(f"Resolved: {result['resolved_text']}")
        print(f"Entities: {result['entities']}")
        
        # Test Polish
        print("\n--- Polish Test ---")
        context = ["Spotkałem wczoraj Janka w parku."]
        text = "On opowiadał mi o swojej nowej pracy."
        
        result = resolver.resolve_coreferences_and_extract_entities(
            text=text,
            context_history=context,
            existing_entities=["Janek", "park"]
        )
        
        print(f"Context: {context}")
        print(f"Original: {text}")
        print(f"Resolved: {result['resolved_text']}")
        print(f"Entities: {result['entities']}")
        
    else:
        print("FastCoref not available")

async def main():
    try:
        await test_simple_case()
        print("\n✅ Test completed successfully!")
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(main())
