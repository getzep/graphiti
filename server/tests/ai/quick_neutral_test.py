"""
Quick test to check if GoEmotions can detect no emotions in neutral texts.
"""
import asyncio
import sys
import os

# Add the parent directory to the path to import modules
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from graph_service.routers.ai.goemotions import extract_emotions_with_goemotions, get_goemotions_detector

async def test_neutral_cases():
    """Test if GoEmotions can detect no emotions in neutral texts."""
    
    neutral_cases = [
        "The meeting is scheduled for 3 PM tomorrow.",
        "Please send me the document by email.",
        "The weather is 25 degrees Celsius.",
        "I went to the store to buy milk.",
        "The file contains 100 pages.",
        "Today is Friday, May 31st.",
        "The car is parked in the garage.",
        "The report shows sales increased by 5%.",
        "",  # Empty string
        "   ",  # Whitespace
        "Yes.",  # Very short
        "OK.",
    ]
    
    print("🧪 Testing GoEmotions with neutral texts")
    print("=" * 50)
    
    detector = get_goemotions_detector()
    
    for i, text in enumerate(neutral_cases, 1):
        print(f"\nTest {i}:")
        print(f"Text: '{text}'")
        
        # Test with different thresholds
        for threshold in [0.3, 0.5, 0.7]:
            emotions = detector.predict_emotions(text, threshold=threshold)
            emoji = "✅" if not emotions else "⚠️"
            print(f"  Threshold {threshold}: {emotions} {emoji}")
    
    print("\n" + "=" * 50)
    print("🔍 Testing edge case with very high threshold")
    
    # Test with very high threshold - should return empty list more often
    test_text = "The meeting is at 3 PM."
    for threshold in [0.8, 0.9, 0.95]:
        emotions = detector.predict_emotions(test_text, threshold=threshold)
        emoji = "✅" if not emotions else "⚠️"
        print(f"Threshold {threshold}: {emotions} {emoji}")

if __name__ == "__main__":
    asyncio.run(test_neutral_cases())
