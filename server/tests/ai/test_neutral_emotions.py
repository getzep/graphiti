"""
Test GoEmotions model's behavior with neutral/emotionally ambiguous text.
This test verifies that the model doesn't hallucinate emotions when they're not present.
"""
import asyncio
import logging
import sys
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Add the project root to the path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

# Import the GoEmotions modules
from graph_service.routers.ai.goemotions import (
    extract_emotions_with_goemotions,
    get_goemotions_detector
)

async def test_neutral_texts():
    """Test GoEmotions with neutral/factual texts that should have minimal emotions."""
    print("\n🧪 Testing GoEmotions with neutral texts...")
    
    # Test cases with neutral/factual content
    neutral_test_cases = [
        "The weather is 25 degrees Celsius today.",
        "The meeting is scheduled for 3 PM.",
        "I went to the store to buy milk.",
        "The report shows sales increased by 5%.",
        "Please send me the document by email.",
        "The car is parked in the garage.",
        "Today is Wednesday, December 13th.",
        "The conference room is on the second floor.",
        "I need to update my password.",
        "The file contains 100 pages.",
    ]
    
    detector = get_goemotions_detector()
    results = []
    
    for i, text in enumerate(neutral_test_cases, 1):
        print(f"\n--- Test {i}/10 ---")
        print(f"Text: '{text}'")
        
        # Test with different thresholds
        for threshold in [0.1, 0.3, 0.5, 0.7]:
            emotions = detector.predict_emotions(text, threshold=threshold)
            print(f"  Threshold {threshold}: {emotions}")
            
            results.append({
                'text': text,
                'threshold': threshold,
                'emotions': emotions,
                'emotion_count': len(emotions)
            })
    
    # Analyze results
    print("\n📊 Analysis:")
    print("=" * 60)
    
    for threshold in [0.1, 0.3, 0.5, 0.7]:
        threshold_results = [r for r in results if r['threshold'] == threshold]
        total_emotions = sum(r['emotion_count'] for r in threshold_results)
        avg_emotions = total_emotions / len(threshold_results)
        texts_with_no_emotions = len([r for r in threshold_results if r['emotion_count'] == 0])
        
        print(f"Threshold {threshold}:")
        print(f"  - Average emotions per text: {avg_emotions:.2f}")
        print(f"  - Texts with no emotions: {texts_with_no_emotions}/10")
        print(f"  - Total emotions detected: {total_emotions}")
    
    return results

async def test_emotionally_ambiguous_texts():
    """Test with texts that could be interpreted as emotional but aren't clearly so."""
    print("\n🤔 Testing GoEmotions with emotionally ambiguous texts...")
    
    ambiguous_test_cases = [
        "I received your message.",
        "The project is finished.",
        "We had a meeting yesterday.",
        "The task took longer than expected.",
        "I'm working on the presentation.",
        "The results are available now.",
        "We need to discuss the budget.",
        "The system is working properly.",
        "I'll check the status tomorrow.",
        "The event starts at 6 PM."
    ]
    
    detector = get_goemotions_detector()
    
    for i, text in enumerate(ambiguous_test_cases, 1):
        print(f"\n--- Ambiguous Test {i}/10 ---")
        print(f"Text: '{text}'")
        
        # Test with standard threshold
        emotions = detector.predict_emotions(text, threshold=0.3)
        print(f"  Emotions (0.3): {emotions}")
        
        # Test with high threshold  
        emotions_strict = detector.predict_emotions(text, threshold=0.7)
        print(f"  Emotions (0.7): {emotions_strict}")

async def test_edge_cases():
    """Test edge cases for emotion detection."""
    print("\n🔍 Testing edge cases...")
    
    edge_cases = [
        "",  # Empty string
        "   ",  # Whitespace only
        "Yes.",  # Very short
        "No.",   # Very short
        "OK.",   # Very short
        "123 456 789",  # Numbers only
        "test test test",  # Repetitive
        "The the the the the.",  # Repetitive words
        "abcdefghijklmnop",  # Random letters
        ".,;:!?",  # Punctuation only
    ]
    
    detector = get_goemotions_detector()
    
    for i, text in enumerate(edge_cases, 1):
        print(f"\n--- Edge Case {i}/10 ---")
        print(f"Text: '{repr(text)}'")
        
        try:
            emotions = detector.predict_emotions(text, threshold=0.3)
            print(f"  Emotions: {emotions}")
        except Exception as e:
            print(f"  Error: {e}")

async def test_with_different_thresholds():
    """Test how threshold affects emotion detection on borderline cases."""
    print("\n⚖️ Testing threshold sensitivity...")
    
    borderline_cases = [
        "I think this might work.",
        "The situation could be better.",
        "We should consider other options.",
        "This is somewhat unusual.",
        "The outcome was as expected.",
    ]
    
    detector = get_goemotions_detector()
    thresholds = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    
    for text in borderline_cases:
        print(f"\nText: '{text}'")
        print("Threshold -> Emotions")
        print("-" * 40)
        
        for threshold in thresholds:
            emotions = detector.predict_emotions(text, threshold=threshold)
            emoji = "🎯" if emotions else "❌"
            print(f"  {threshold:3.1f}     -> {emotions} {emoji}")

async def test_confidence_scores():
    """Test to see actual confidence scores for neutral texts."""
    print("\n📈 Testing confidence scores for neutral texts...")
    
    import torch
    
    detector = get_goemotions_detector()
    neutral_texts = [
        "The meeting is at 3 PM.",
        "I went to the store.",
        "The weather is nice today."
    ]
    
    for text in neutral_texts:
        print(f"\nText: '{text}'")
        print("Emotion -> Confidence")
        print("-" * 30)
        
        try:
            # Get raw predictions
            inputs = detector.tokenizer(
                text,
                return_tensors="pt",
                truncation=True,
                padding=True,
                max_length=512
            )
            inputs = {k: v.to(detector.device) for k, v in inputs.items()}
            
            with torch.no_grad():
                outputs = detector.model(**inputs)
                predictions = torch.sigmoid(outputs.logits)
            
            predictions_cpu = predictions.cpu().numpy()[0]
            
            # Show top 5 emotions with scores
            emotion_scores = list(zip(detector.EMOTION_LABELS, predictions_cpu))
            emotion_scores.sort(key=lambda x: x[1], reverse=True)
            
            for emotion, score in emotion_scores[:5]:
                print(f"  {emotion:15} -> {score:.4f}")
                
        except Exception as e:
            print(f"  Error: {e}")

async def main():
    """Run all neutral emotion tests."""
    print("🧠 GoEmotions Neutral Text Testing")
    print("=" * 50)
    
    try:
        # Run all test suites
        await test_neutral_texts()
        await test_emotionally_ambiguous_texts()
        await test_edge_cases()
        await test_with_different_thresholds()
        await test_confidence_scores()
        
        print("\n✅ All neutral emotion tests completed!")
        print("\n💡 Key Insights:")
        print("- Check if higher thresholds reduce false positive emotions")
        print("- Verify that truly neutral texts don't trigger emotions")
        print("- Ensure edge cases are handled gracefully")
        
        return True
        
    except Exception as e:
        print(f"\n❌ Testing failed: {e}")
        logger.error(f"Testing failed: {e}")
        return False

if __name__ == "__main__":
    success = asyncio.run(main())
    sys.exit(0 if success else 1)
