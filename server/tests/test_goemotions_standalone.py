"""
Test GoEmotions integration without requiring OpenAI API.
This test focuses specifically on the GoEmotions emotion detection functionality.
"""
import asyncio
import logging
import sys
import os

# Add the parent directory to the path so we can import modules
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

async def test_goemotions_standalone():
    """Test GoEmotions model functionality without OpenAI dependency."""
    logger.info("Testing GoEmotions standalone functionality...")
    
    try:
        # Import GoEmotions modules
        from graph_service.routers.ai.goemotions import (
            get_goemotions_detector, 
            extract_emotions_with_goemotions
        )
        
        # Test 1: Model loading
        logger.info("1. Testing model loading...")
        detector = get_goemotions_detector()
        assert detector is not None, "GoEmotions detector should be available"
        assert detector.model is not None, "GoEmotions model should be loaded"
        assert detector.tokenizer is not None, "GoEmotions tokenizer should be loaded"
        logger.info("✅ Model loaded successfully")
        
        # Test 2: Basic emotion detection
        logger.info("2. Testing basic emotion detection...")
        test_texts = [
            "I am so excited about this new project!",
            "This makes me really sad and disappointed.",
            "I'm feeling very anxious about the presentation tomorrow.",
            "Thank you so much! I really appreciate your help.",
            "This is just a neutral statement without much emotion."
        ]
        
        for i, text in enumerate(test_texts):
            emotions = detector.predict_emotions(text, threshold=0.3)
            logger.info(f"Text {i+1}: '{text[:50]}...'")
            logger.info(f"Detected emotions: {emotions}")
            assert isinstance(emotions, list), "Emotions should be a list"
            # Note: Even neutral text might detect some emotion, so we don't require len > 0
        
        logger.info("✅ Basic emotion detection working")
        
        # Test 3: Async wrapper function
        logger.info("3. Testing async wrapper function...")
        text = "I'm really happy and grateful for this opportunity, but also a bit nervous."
        existing_emotions = ["joy", "anxiety"]
        
        emotions = await extract_emotions_with_goemotions(text, existing_emotions)
        logger.info(f"Async extraction result: {emotions}")
        assert isinstance(emotions, list), "Async function should return a list"
        logger.info("✅ Async wrapper working")
        
        # Test 4: Emotion mapping
        logger.info("4. Testing emotion mapping...")
        detected_emotions = ["excitement", "nervousness", "gratitude"]
        existing_emotions = ["enthusiasm", "anxiety", "thankfulness"]
        
        mapped_emotions = detector.map_to_existing_emotions(detected_emotions, existing_emotions)
        logger.info(f"Original: {detected_emotions}")
        logger.info(f"Existing: {existing_emotions}")
        logger.info(f"Mapped: {mapped_emotions}")
        assert isinstance(mapped_emotions, list), "Mapped emotions should be a list"
        logger.info("✅ Emotion mapping working")
        
        # Test 5: Edge cases
        logger.info("5. Testing edge cases...")
        
        # Empty text
        emotions = await extract_emotions_with_goemotions("", [])
        assert emotions == [], "Empty text should return empty list"
        
        # Whitespace only
        emotions = await extract_emotions_with_goemotions("   \n\t   ", [])
        assert emotions == [], "Whitespace only should return empty list"
        
        # Very long text (should be truncated)
        long_text = "I am happy. " * 200  # Very long text
        emotions = await extract_emotions_with_goemotions(long_text, [])
        assert isinstance(emotions, list), "Long text should still work"
        logger.info(f"Long text emotions: {emotions}")
        
        logger.info("✅ Edge cases handled correctly")
        
        # Test 6: Multiple emotion detection
        logger.info("6. Testing complex emotional text...")
        complex_text = "I was initially scared and worried about the interview, but when I got the job offer, I became incredibly excited and grateful. Now I'm feeling proud and optimistic about the future."
        emotions = await extract_emotions_with_goemotions(complex_text, [])
        logger.info(f"Complex text: '{complex_text[:50]}...'")
        logger.info(f"Detected emotions: {emotions}")
        
        # Should detect multiple emotions
        assert isinstance(emotions, list), "Should return a list"
        assert len(emotions) >= 1, "Complex emotional text should detect at least one emotion"
        
        # Check for expected emotion categories
        emotion_set = set(emotions)
        negative_emotions = {"fear", "anxiety", "nervousness", "worry", "scared"}
        positive_emotions = {"excitement", "gratitude", "pride", "optimism", "joy", "happy"}
        
        has_negative = len(emotion_set.intersection(negative_emotions)) > 0
        has_positive = len(emotion_set.intersection(positive_emotions)) > 0
        
        logger.info(f"Detected negative emotions: {has_negative}")
        logger.info(f"Detected positive emotions: {has_positive}")
        
        # At least one type should be detected
        assert has_negative or has_positive, f"Expected to detect emotional content, got: {emotions}"
        
        logger.info("✅ Complex emotion detection working")
        
        return True
        
    except ImportError as e:
        logger.error(f"❌ Import error: {e}")
        logger.error("Make sure you're running from the correct directory and dependencies are installed")
        return False
    except Exception as e:
        logger.error(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

async def main():
    """Run the GoEmotions standalone test."""
    logger.info("=" * 60)
    logger.info("GoEmotions Standalone Integration Test")
    logger.info("=" * 60)
    
    success = await test_goemotions_standalone()
    
    if success:
        logger.info("=" * 60)
        logger.info("🎉 All GoEmotions tests passed successfully!")
        logger.info("The GoEmotions model is properly integrated and working.")
        logger.info("=" * 60)
    else:
        logger.error("=" * 60)
        logger.error("❌ GoEmotions tests failed!")
        logger.error("=" * 60)
        sys.exit(1)

if __name__ == "__main__":
    asyncio.run(main())
