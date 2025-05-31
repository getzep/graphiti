"""
Test the integration of GoEmotions with the server's extraction pipeline.
This test verifies that the emotion extraction works end-to-end in the server context.
"""
import pytest
import asyncio
import logging
import os
import sys
from typing import List, Dict, Any

# Add the server directory to the path to import modules
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Import the extraction modules
from graph_service.routers.ai.extraction import (
    extract_facts_emotions_entities,
    extract_facts_and_emotions_with_openai,
    extract_emotions_with_goemotions
)
from graph_service.routers.ai.goemotions import get_goemotions_detector

class TestExtractionPipelineIntegration:
    """Test GoEmotions integration with the extraction pipeline."""
    
    def setup_method(self):
        """Setup for each test method."""
        logger.info("Setting up test...")
    
    @pytest.mark.asyncio
    async def test_goemotions_direct_integration(self):
        """Test GoEmotions model directly."""
        logger.info("Testing GoEmotions direct integration...")
        
        test_text = "I am so excited about this new project! It makes me feel really happy and optimistic."
        existing_emotions = ["joy", "enthusiasm"]
        
        # Test direct GoEmotions integration
        emotions = await extract_emotions_with_goemotions(test_text, existing_emotions)
        
        assert isinstance(emotions, list), "Emotions should be a list"
        assert len(emotions) > 0, "Should detect at least one emotion"
        
        logger.info(f"Detected emotions: {emotions}")
          # Verify we get reasonable emotions
        expected_emotions = ["excitement", "joy", "optimism", "happiness", "enthusiasm", "admiration"]
        detected_emotion_types = set(emotions)
        expected_emotion_types = set(expected_emotions)
        
        # Should have some overlap with expected emotions
        overlap = detected_emotion_types.intersection(expected_emotion_types)
        assert len(overlap) > 0, f"Expected some overlap between detected {emotions} and expected {expected_emotions}"
    
    @pytest.mark.asyncio
    async def test_facts_and_emotions_extraction(self):
        """Test the combined facts and emotions extraction."""
        logger.info("Testing combined facts and emotions extraction...")
        
        test_text = "John met Sarah at the coffee shop yesterday. He was very nervous but also excited about their first date."
        existing_emotions = ["anxiety", "happiness"]
        
        # Test the OpenAI + GoEmotions integration
        result = await extract_facts_and_emotions_with_openai(
            test_text,
            chat_history=None,
            existing_emotions=existing_emotions
        )
        
        assert isinstance(result, dict), "Result should be a dictionary"
        assert "facts" in result, "Result should contain facts"
        assert "emotions" in result, "Result should contain emotions"
        assert "usage" in result, "Result should contain usage information"
        
        facts = result["facts"]
        emotions = result["emotions"]
        
        logger.info(f"Extracted facts: {facts}")
        logger.info(f"Extracted emotions: {emotions}")
        
        # Verify emotions are detected
        assert isinstance(emotions, list), "Emotions should be a list"
        assert len(emotions) > 0, "Should detect emotions from emotional text"
        
        # Should detect nervousness or excitement
        emotion_set = set(emotions)
        expected_emotions = {"nervousness", "excitement", "anxiety", "nervous"}
        overlap = emotion_set.intersection(expected_emotions)
        assert len(overlap) > 0, f"Expected to detect nervousness or excitement, got: {emotions}"
    
    @pytest.mark.asyncio
    async def test_full_pipeline_integration(self):
        """Test the full extraction pipeline including entities, facts, and emotions."""
        logger.info("Testing full extraction pipeline integration...")
        
        test_text = "Alice told me she was feeling overwhelmed with work but excited about the weekend plans with Bob."
        existing_entities = ["Alice", "Bob"]
        existing_emotions = ["stress", "joy"]
        chat_history = [
            {"role": "user", "content": "I spoke with Alice yesterday"},
            {"role": "user", "content": "She mentioned Bob was coming to visit"}
        ]
        
        # Test full pipeline
        result = await extract_facts_emotions_entities(
            message_content=test_text,
            existing_emotions=existing_emotions,
            existing_entities=existing_entities,
            chat_history=chat_history
        )
        
        assert isinstance(result, dict), "Result should be a dictionary"
        
        # Check all expected components
        required_keys = ["facts", "emotions", "entities", "resolved_text", "usage", "coreference_info"]
        for key in required_keys:
            assert key in result, f"Result should contain {key}"
        
        facts = result["facts"]
        emotions = result["emotions"]
        entities = result["entities"]
        resolved_text = result["resolved_text"]
        
        logger.info(f"Extracted facts: {facts}")
        logger.info(f"Extracted emotions: {emotions}")
        logger.info(f"Extracted entities: {entities}")
        logger.info(f"Resolved text: {resolved_text}")
        
        # Verify emotions are extracted
        assert isinstance(emotions, list), "Emotions should be a list"
        assert len(emotions) > 0, "Should detect emotions"
        
        # Should detect some form of overwhelm/stress and excitement
        emotion_set = set(emotions)
        stress_emotions = {"overwhelm", "stress", "anxiety", "pressure"}
        positive_emotions = {"excitement", "joy", "optimism", "enthusiasm"}
        
        has_stress = len(emotion_set.intersection(stress_emotions)) > 0
        has_positive = len(emotion_set.intersection(positive_emotions)) > 0
        
        assert has_stress or has_positive, f"Expected to detect stress or positive emotions, got: {emotions}"
        
        # Verify entities are preserved/extracted
        assert isinstance(entities, list), "Entities should be a list"
        entity_set = set(entities)
        assert "Alice" in entity_set or "Bob" in entity_set, f"Should preserve/extract entities, got: {entities}"
    
    @pytest.mark.asyncio
    async def test_goemotions_model_availability(self):
        """Test that GoEmotions model loads correctly."""
        logger.info("Testing GoEmotions model availability...")
        
        detector = get_goemotions_detector()
        
        assert detector is not None, "GoEmotions detector should be available"
        assert detector.model is not None, "GoEmotions model should be loaded"
        assert detector.tokenizer is not None, "GoEmotions tokenizer should be loaded"
        
        # Test prediction capability
        test_text = "This is a simple test message."
        emotions = detector.predict_emotions(test_text)
        
        assert isinstance(emotions, list), "Predictions should return a list"
        logger.info(f"Model successfully predicted emotions: {emotions}")
    
    @pytest.mark.asyncio
    async def test_emotion_mapping(self):
        """Test emotion mapping to existing emotions."""
        logger.info("Testing emotion mapping...")
        
        detector = get_goemotions_detector()
        
        # Test mapping with existing emotions
        detected_emotions = ["excitement", "nervousness", "joy"]
        existing_emotions = ["enthusiasm", "anxiety", "happiness"]
        
        mapped_emotions = detector.map_to_existing_emotions(detected_emotions, existing_emotions)
        
        assert isinstance(mapped_emotions, list), "Mapped emotions should be a list"
        
        logger.info(f"Original emotions: {detected_emotions}")
        logger.info(f"Existing emotions: {existing_emotions}")
        logger.info(f"Mapped emotions: {mapped_emotions}")
        
        # Should map similar emotions
        mapped_set = set(mapped_emotions)
        # excitement -> enthusiasm, nervousness -> anxiety, joy -> happiness
        expected_mappings = {"enthusiasm", "anxiety", "happiness"}
        
        # At least one mapping should occur
        overlap = mapped_set.intersection(expected_mappings)
        assert len(overlap) > 0, f"Expected some emotion mapping, got: {mapped_emotions}"
    
    @pytest.mark.asyncio
    async def test_empty_text_handling(self):
        """Test handling of empty or invalid text."""
        logger.info("Testing empty text handling...")
        
        # Test with empty text
        result = await extract_facts_emotions_entities(
            message_content="",
            existing_emotions=[],
            existing_entities=[],
            chat_history=None
        )
        
        assert isinstance(result, dict), "Result should be a dictionary"
        assert result["facts"] == [], "Facts should be empty for empty text"
        assert result["emotions"] == [], "Emotions should be empty for empty text"
        assert result["entities"] == [], "Entities should be empty for empty text"
        
        # Test with whitespace only
        result = await extract_facts_emotions_entities(
            message_content="   \n\t   ",
            existing_emotions=[],
            existing_entities=[],
            chat_history=None
        )
        
        assert isinstance(result, dict), "Result should be a dictionary"
        assert result["facts"] == [], "Facts should be empty for whitespace text"
        assert result["emotions"] == [], "Emotions should be empty for whitespace text"

if __name__ == "__main__":
    async def run_tests():
        test_instance = TestExtractionPipelineIntegration()
        
        # Run all tests
        tests = [
            test_instance.test_goemotions_model_availability(),
            test_instance.test_goemotions_direct_integration(),
            test_instance.test_emotion_mapping(),
            test_instance.test_facts_and_emotions_extraction(),
            test_instance.test_full_pipeline_integration(),
            test_instance.test_empty_text_handling(),
        ]
        
        for i, test in enumerate(tests):
            try:
                logger.info(f"\n{'='*60}")
                logger.info(f"Running test {i+1}/{len(tests)}")
                await test
                logger.info(f"✅ Test {i+1} passed")
            except Exception as e:
                logger.error(f"❌ Test {i+1} failed: {e}")
                raise
        
        logger.info(f"\n{'='*60}")
        logger.info("🎉 All tests completed successfully!")
    
    # Run the tests
    asyncio.run(run_tests())
