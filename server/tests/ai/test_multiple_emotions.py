"""
Test to check if GoEmotions can detect multiple emotions in a single text.
"""
import asyncio
import sys
import os

# Add the server directory to the path to import modules
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from graph_service.routers.ai.goemotions import extract_emotions_with_goemotions, get_goemotions_detector

async def test_multiple_emotions():
    """Test if GoEmotions can detect multiple emotions in a single text."""
    
    multi_emotion_cases = [
        {
            "text": "I'm excited about the promotion but nervous about the new responsibilities.",
            "expected": ["excitement", "nervousness", "joy", "fear", "anxiety"]
        },
        {
            "text": "I'm grateful for your help but frustrated that I needed it in the first place.",
            "expected": ["gratitude", "frustration", "annoyance"]
        },
        {
            "text": "I love this job, but I'm disappointed with my salary and angry about the lack of recognition.",
            "expected": ["love", "disappointment", "anger", "sadness"]
        },
        {
            "text": "The movie was amazing and hilarious, but the ending made me cry with sadness.",
            "expected": ["joy", "amusement", "sadness", "admiration"]
        },
        {
            "text": "I'm proud of my achievement but also embarrassed by all the attention.",
            "expected": ["pride", "embarrassment"]
        },
        {
            "text": "I'm curious about the results but afraid of what they might reveal.",
            "expected": ["curiosity", "fear"]
        },
        {
            "text": "I feel relieved that it's over, grateful for the support, but sad that this chapter is ending.",
            "expected": ["relief", "gratitude", "sadness"]
        }
    ]
    
    print("🧪 Testing GoEmotions with multi-emotion texts")
    print("=" * 60)
    
    detector = get_goemotions_detector()
    
    for i, case in enumerate(multi_emotion_cases, 1):
        text = case["text"]
        expected = case["expected"]
        
        print(f"\nTest {i}:")
        print(f"Text: '{text}'")
        print(f"Expected emotions: {expected}")
        
        # Test with different thresholds
        for threshold in [0.1, 0.3, 0.5]:
            emotions = detector.predict_emotions(text, threshold=threshold)
            emoji = "🎯" if len(emotions) > 1 else ("✅" if len(emotions) == 1 else "❌")
            
            # Check if any expected emotions were found
            found_expected = [e for e in emotions if e in expected]
            
            print(f"  Threshold {threshold}: {emotions} {emoji}")
            if found_expected:
                print(f"    ✅ Found expected: {found_expected}")
            if len(emotions) > 1:
                print(f"    🎉 Multiple emotions detected: {len(emotions)} emotions")
    
    print("\n" + "=" * 60)
    print("📊 Summary: Testing emotion count distribution")
    
    # Test a variety of texts and see emotion count distribution
    all_results = []
    
    for case in multi_emotion_cases:
        emotions = detector.predict_emotions(case["text"], threshold=0.3)
        all_results.append({
            "text": case["text"][:50] + "...",
            "emotion_count": len(emotions),
            "emotions": emotions
        })
    
    # Count distribution
    emotion_counts = {}
    for result in all_results:
        count = result["emotion_count"]
        emotion_counts[count] = emotion_counts.get(count, 0) + 1
    
    print(f"\nEmotion count distribution (threshold 0.3):")
    for count in sorted(emotion_counts.keys()):
        print(f"  {count} emotion(s): {emotion_counts[count]} texts")
    
    # Show examples with multiple emotions
    multi_emotion_results = [r for r in all_results if r["emotion_count"] > 1]
    if multi_emotion_results:
        print(f"\n🎯 Texts with multiple emotions ({len(multi_emotion_results)}/{len(all_results)}):")
        for result in multi_emotion_results:
            print(f"  '{result['text']}' -> {result['emotions']}")
    else:
        print(f"\n⚠️  No texts detected multiple emotions")

async def test_confidence_scores_multiple():
    """Show confidence scores for texts that should have multiple emotions."""
    print("\n" + "=" * 60)
    print("📈 Analyzing confidence scores for multi-emotion detection")
    
    import torch
    
    detector = get_goemotions_detector()
    test_text = "I'm excited about the promotion but nervous about the responsibilities."
    
    print(f"\nText: '{test_text}'")
    print("Emotion -> Confidence")
    print("-" * 40)
    
    try:
        # Get raw predictions
        inputs = detector.tokenizer(
            test_text,
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
        
        # Show all emotions with scores above 0.05
        emotion_scores = list(zip(detector.EMOTION_LABELS, predictions_cpu))
        emotion_scores.sort(key=lambda x: x[1], reverse=True)
        
        threshold_counts = {}
        
        for emotion, score in emotion_scores:
            if score > 0.05:  # Only show meaningful scores
                # Check which thresholds this would pass
                thresholds_passed = []
                for t in [0.1, 0.2, 0.3, 0.4, 0.5]:
                    if score > t:
                        thresholds_passed.append(f"{t}")
                
                threshold_str = " | ".join(thresholds_passed) if thresholds_passed else "none"
                print(f"  {emotion:15} -> {score:.4f} (passes: {threshold_str})")
                
                # Count for each threshold
                for t in [0.1, 0.2, 0.3, 0.4, 0.5]:
                    if score > t:
                        if t not in threshold_counts:
                            threshold_counts[t] = 0
                        threshold_counts[t] += 1
        
        print(f"\nEmotion count by threshold:")
        for threshold in sorted(threshold_counts.keys()):
            count = threshold_counts[threshold]
            print(f"  Threshold {threshold}: {count} emotions")
            
    except Exception as e:
        print(f"  Error: {e}")

if __name__ == "__main__":
    asyncio.run(test_multiple_emotions())
    asyncio.run(test_confidence_scores_multiple())
