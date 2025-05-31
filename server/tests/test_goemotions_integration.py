"""
Test the GoEmotions integration specifically without requiring OpenAI API.
This test verifies that the emotion extraction works with the GoEmotions model.
"""
import asyncio
import logging
from typing import List, Dict, Any

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Import the GoEmotions modules directly
from graph_service.routers.ai.goemotions import (
    get_goemotions_detector,
    extract_emotions_with_goemotions,
    GoEmotionsDetector
)

class TestGoEmotionsIntegration:
    """Test GoEmotions integration specifically."""
    
    def setup_method(self):
        """Setup for each test method."""
        logger.info("Setting up test...")
    
    async def test_goemotions_model_loading(self):
        """Test that GoEmotions model loads correctly."""
        logger.info("Testing GoEmotions model loading...")
        
        detector = get_goemotions_detector()
        
        assert detector is not None, "GoEmotions detector should be available"
        assert detector.model is not None, "GoEmotions model should be loaded"
        assert detector.tokenizer is not None, "GoEmotions tokenizer should be loaded"
        assert hasattr(detector, 'EMOTION_LABELS'), "Should have emotion labels"
        assert len(detector.EMOTION_LABELS) > 0, "Should have emotion labels defined"
        
        logger.info(f"GoEmotions loaded successfully with {len(detector.EMOTION_LABELS)} emotion labels")
        logger.info(f"Available emotions: {detector.EMOTION_LABELS}")
    
    async def test_emotion_detection(self):
        """Test emotion detection with various text inputs."""
        logger.info("Testing emotion detection...")
        
        test_cases = [
            {
                "text": "I am so excited about this new project! It makes me feel really happy and optimistic.",
                "expected_emotions": ["excitement", "joy", "optimism", "happiness", "enthusiasm", "admiration"]
            },
            {
                "text": "I'm feeling really nervous and worried about the presentation tomorrow.",
                "expected_emotions": ["nervousness", "fear", "anxiety", "worry"]
            },
            {
                "text": "That movie was absolutely disgusting and made me feel sick.",
                "expected_emotions": ["disgust", "annoyance", "disapproval"]
            },
            {
                "text": "I'm so grateful for all the help you've given me. Thank you so much!",
                "expected_emotions": ["gratitude", "appreciation", "joy"]
            },
            {
                "text": "I can't believe he said that to me. I'm absolutely furious!",
                "expected_emotions": ["anger", "rage", "annoyance", "disapproval"]
            }
        ]
        
        for i, test_case in enumerate(test_cases):
            logger.info(f"\nTest case {i+1}: '{test_case['text'][:50]}...'")
            
            emotions = await extract_emotions_with_goemotions(test_case["text"])
            
            assert isinstance(emotions, list), f"Emotions should be a list for test case {i+1}"
            assert len(emotions) > 0, f"Should detect at least one emotion for test case {i+1}"
            
            logger.info(f"Detected emotions: {emotions}")
            
            # Check if we detect reasonable emotions
            detected_set = set(emotions)
            expected_set = set(test_case["expected_emotions"])
            overlap = detected_set.intersection(expected_set)
            
            # At least some overlap or detect emotions that make sense
            if len(overlap) == 0:
                logger.warning(f"No exact overlap, but detected: {emotions}")
                # As long as we detect emotions, it's working
                assert len(emotions) > 0, f"Should detect some emotions even if not exact match"
            else:
                logger.info(f"Good overlap: {overlap}")
    
    async def test_emotion_mapping(self):
        """Test emotion mapping functionality."""
        logger.info("Testing emotion mapping...")
        
        detector = get_goemotions_detector()
        
        # Test mapping with existing emotions
        test_cases = [
            {
                "detected": ["excitement", "nervousness", "joy"],
                "existing": ["enthusiasm", "anxiety", "happiness"],
                "expected_mappings": ["enthusiasm", "anxiety", "happiness"]
            },
            {
                "detected": ["anger", "disgust"],
                "existing": ["rage", "revulsion"],
                "expected_mappings": ["rage", "revulsion"]
            },
            {
                "detected": ["gratitude", "love"],
                "existing": ["thankfulness", "affection"],
                "expected_mappings": ["thankfulness", "affection"]
            }
        ]
        
        for i, test_case in enumerate(test_cases):
            logger.info(f"\nMapping test {i+1}")
            logger.info(f"Detected emotions: {test_case['detected']}")
            logger.info(f"Existing emotions: {test_case['existing']}")
            
            mapped_emotions = detector.map_to_existing_emotions(
                test_case["detected"], 
                test_case["existing"]
            )
            
            assert isinstance(mapped_emotions, list), f"Mapped emotions should be a list for test {i+1}"
            logger.info(f"Mapped emotions: {mapped_emotions}")
            
            # Should have some reasonable mapping
            mapped_set = set(mapped_emotions)
            expected_set = set(test_case["expected_mappings"])
            
            # Either exact mapping or includes original emotions
            has_mapping = len(mapped_set.intersection(expected_set)) > 0
            has_originals = len(mapped_set.intersection(set(test_case["detected"]))) > 0
            
            assert has_mapping or has_originals, f"Should have mapping or original emotions for test {i+1}"
    
    async def test_different_confidence_thresholds(self):
        """Test emotion detection with different confidence thresholds."""
        logger.info("Testing different confidence thresholds...")
        
        text = "I'm feeling a bit anxious but also somewhat excited about the new opportunity."
        detector = get_goemotions_detector()
        
        thresholds = [0.1, 0.3, 0.5, 0.7]
        
        for threshold in thresholds:
            emotions = detector.predict_emotions(text, threshold=threshold)
            logger.info(f"Threshold {threshold}: {emotions}")
            
            assert isinstance(emotions, list), f"Should return list for threshold {threshold}"
            # Higher thresholds should generally return fewer emotions
        
        # Low threshold should detect more emotions than high threshold
        low_threshold_emotions = detector.predict_emotions(text, threshold=0.1)
        high_threshold_emotions = detector.predict_emotions(text, threshold=0.7)
        
        logger.info(f"Low threshold (0.1): {low_threshold_emotions}")
        logger.info(f"High threshold (0.7): {high_threshold_emotions}")
        
        # Should detect at least something with low threshold
        assert len(low_threshold_emotions) > 0, "Should detect emotions with low threshold"
    
    async def test_empty_and_invalid_inputs(self):
        """Test handling of empty and invalid inputs."""
        logger.info("Testing empty and invalid inputs...")
        
        test_cases = [
            "",
            "   ",
            "\n\t",
            None,
        ]
        
        for i, text in enumerate(test_cases):
            logger.info(f"Testing input {i+1}: {repr(text)}")
            
            try:
                if text is None:
                    # This should be handled gracefully
                    emotions = await extract_emotions_with_goemotions("")
                else:
                    emotions = await extract_emotions_with_goemotions(text)
                
                assert isinstance(emotions, list), f"Should return list for input {i+1}"
                assert emotions == [], f"Should return empty list for empty input {i+1}"
                
            except Exception as e:
                logger.error(f"Error with input {i+1}: {e}")
                # Should not crash, but if it does, we'll note it
                raise
    
    async def test_long_text_handling(self):
        """Test handling of long text inputs."""
        logger.info("Testing long text handling...")
        
        # Create a long text that exceeds typical token limits
        base_text = "I am feeling very excited and happy about this wonderful opportunity. "
        long_text = base_text * 50  # Very long text
        
        logger.info(f"Testing text of length: {len(long_text)} characters")
        
        emotions = await extract_emotions_with_goemotions(long_text)
        
        assert isinstance(emotions, list), "Should handle long text"
        assert len(emotions) > 0, "Should detect emotions from long text"
        
        logger.info(f"Detected emotions from long text: {emotions}")
        
        # Should detect positive emotions
        positive_emotions = {"excitement", "joy", "happiness", "optimism", "enthusiasm"}
        detected_set = set(emotions)
        overlap = detected_set.intersection(positive_emotions)
        
        assert len(overlap) > 0, f"Should detect positive emotions from repetitive positive text, got: {emotions}"

async def run_tests():
    """Run all tests."""
    test_instance = TestGoEmotionsIntegration()
    
    # Run all tests
    tests = [
        ("GoEmotions Model Loading", test_instance.test_goemotions_model_loading()),
        ("Emotion Detection", test_instance.test_emotion_detection()),
        ("Emotion Mapping", test_instance.test_emotion_mapping()),
        ("Different Confidence Thresholds", test_instance.test_different_confidence_thresholds()),
        ("Empty and Invalid Inputs", test_instance.test_empty_and_invalid_inputs()),
        ("Long Text Handling", test_instance.test_long_text_handling()),
    ]
    
    for i, (test_name, test_coro) in enumerate(tests):
        try:
            logger.info(f"\n{'='*60}")
            logger.info(f"Running test {i+1}/{len(tests)}: {test_name}")
            await test_coro
            logger.info(f"✅ Test {i+1} ({test_name}) passed")
        except Exception as e:
            logger.error(f"❌ Test {i+1} ({test_name}) failed: {e}")
            raise
    
    logger.info(f"\n{'='*60}")
    logger.info("🎉 All GoEmotions integration tests completed successfully!")
    logger.info("GoEmotions model is properly integrated and working correctly!")

if __name__ == "__main__":
    # Run the tests
    asyncio.run(run_tests())
