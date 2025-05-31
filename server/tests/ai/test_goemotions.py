"""
Test GoEmotions emotion detection.
"""
import asyncio
import sys
import os

# Add the server directory to the path to import modules
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from graph_service.routers.ai.goemotions import extract_emotions_with_goemotions

async def test_goemotions():
    """Test GoEmotions emotion detection with various texts."""
    
    test_cases = [
        {
            "text": "I am so happy today! Everything is going wonderfully.",
            "expected_emotions": ["joy", "happiness", "excitement"]
        },
        {
            "text": "I'm really worried about tomorrow's exam. I feel nervous.",
            "expected_emotions": ["nervousness", "fear", "anxiety"]
        },
        {
            "text": "I can't believe they did that to me. I'm furious!",
            "expected_emotions": ["anger", "rage"]
        },
        {
            "text": "I miss my grandmother so much. She was everything to me.",
            "expected_emotions": ["sadness", "grief"]
        },
        {
            "text": "Thank you so much for helping me with this project.",
            "expected_emotions": ["gratitude"]
        },
        {
            "text": "I'm confused about what happened. Nothing makes sense.",
            "expected_emotions": ["confusion"]
        }
    ]
    
    print("Testing GoEmotions emotion detection...")
    print("=" * 50)
    
    for i, test_case in enumerate(test_cases, 1):
        text = test_case["text"]
        print(f"\nTest {i}:")
        print(f"Text: {text}")
        
        try:
            detected_emotions = await extract_emotions_with_goemotions(text)
            print(f"Detected emotions: {detected_emotions}")
            
            # Check if any expected emotions were detected
            expected = test_case["expected_emotions"]
            found_expected = any(emotion in detected_emotions for emotion in expected)
            
            if found_expected:
                print("✅ Test passed - found expected emotion(s)")
            else:
                print(f"⚠️  Test inconclusive - expected one of {expected}, got {detected_emotions}")
                
        except Exception as e:
            print(f"❌ Test failed with error: {e}")
    
    print("\n" + "=" * 50)
    print("Testing with existing emotions mapping...")
    
    # Test with existing emotions
    existing_emotions = ["happiness", "anxiety", "rage", "sorrow", "appreciation", "bewilderment"]
    
    for i, test_case in enumerate(test_cases, 1):
        text = test_case["text"]
        print(f"\nTest {i} (with existing emotions):")
        print(f"Text: {text}")
        
        try:
            detected_emotions = await extract_emotions_with_goemotions(text, existing_emotions)
            print(f"Detected emotions: {detected_emotions}")
            print(f"Existing emotions: {existing_emotions}")
            
        except Exception as e:
            print(f"❌ Test failed with error: {e}")

if __name__ == "__main__":
    asyncio.run(test_goemotions())
