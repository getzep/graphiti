"""
GoEmotions model for emotion detection.
This module provides emotion detection using the GoEmotions model instead of OpenAI.
"""
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from typing import List, Dict, Any
import logging

logger = logging.getLogger(__name__)

class GoEmotionsDetector:
    """GoEmotions emotion detector."""
    
    # GoEmotions emotion labels
    EMOTION_LABELS = [
        'admiration', 'amusement', 'anger', 'annoyance', 'approval', 'caring', 'confusion',
        'curiosity', 'desire', 'disappointment', 'disapproval', 'disgust', 'embarrassment',
        'excitement', 'fear', 'gratitude', 'grief', 'joy', 'love', 'nervousness', 'optimism',
        'pride', 'realization', 'relief', 'remorse', 'sadness', 'surprise', 'neutral'
    ]
    
    def __init__(self):
        self.model = None
        self.tokenizer = None
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self._load_model()
    
    def _load_model(self):
        """Load the GoEmotions model and tokenizer."""
        try:
            model_name = "monologg/bert-base-cased-goemotions-original"
            logger.info(f"Loading GoEmotions model: {model_name}")
            
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.model = AutoModelForSequenceClassification.from_pretrained(model_name)
            self.model.to(self.device)
            self.model.eval()
            
            logger.info("GoEmotions model loaded successfully")
        except Exception as e:
            logger.error(f"Error loading GoEmotions model: {e}")
            raise
    
    def predict_emotions(self, text: str, threshold: float = 0.3) -> List[str]:
        """
        Predict emotions from text using GoEmotions model.
        
        Args:
            text: Input text to analyze
            threshold: Minimum confidence threshold for emotion detection
            
        Returns:
            List of detected emotion labels
        """
        if not text or not text.strip():
            return []
        
        try:
            # Tokenize input
            inputs = self.tokenizer(
                text,
                return_tensors="pt",
                truncation=True,
                padding=True,
                max_length=512
            )
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            # Get predictions
            with torch.no_grad():
                outputs = self.model(**inputs)
                predictions = torch.sigmoid(outputs.logits)
            
            # Extract emotions above threshold
            detected_emotions = []
            predictions_cpu = predictions.cpu().numpy()[0]
            
            for i, confidence in enumerate(predictions_cpu):
                if confidence > threshold:
                    emotion = self.EMOTION_LABELS[i]
                    if emotion != 'neutral':  # Skip neutral emotion
                        detected_emotions.append(emotion)
              # If no emotions detected above threshold, check if we should use neutral threshold
            if not detected_emotions:
                # Check if neutral emotion is the highest scoring and above a minimum confidence
                neutral_idx = self.EMOTION_LABELS.index('neutral') if 'neutral' in self.EMOTION_LABELS else -1
                if neutral_idx >= 0:
                    neutral_confidence = predictions_cpu[neutral_idx]
                    
                    # If neutral is high confidence (>0.5), return empty list (no emotions)
                    if neutral_confidence > 0.5:
                        logger.debug(f"High neutral confidence ({neutral_confidence:.3f}), returning no emotions")
                        return []
                
                # Only use fallback if no strong neutral signal and threshold is not too high
                if threshold < 0.7:  # Only fallback for reasonable thresholds
                    non_neutral_indices = [i for i, label in enumerate(self.EMOTION_LABELS) if label != 'neutral']
                    if non_neutral_indices:
                        non_neutral_predictions = [predictions_cpu[i] for i in non_neutral_indices]
                        max_confidence = max(non_neutral_predictions)
                        
                        # Only use fallback if the highest emotion has reasonable confidence
                        if max_confidence > 0.1:  # Minimum fallback confidence
                            max_idx = non_neutral_indices[non_neutral_predictions.index(max_confidence)]
                            detected_emotions.append(self.EMOTION_LABELS[max_idx])
                            logger.debug(f"Fallback emotion: {self.EMOTION_LABELS[max_idx]} (confidence: {max_confidence:.3f})")
                        else:
                            logger.debug(f"All emotions below minimum confidence ({max_confidence:.3f}), returning no emotions")
                else:
                    logger.debug(f"High threshold ({threshold}), no fallback used")
            
            logger.debug(f"Detected emotions for text '{text[:50]}...': {detected_emotions}")
            return detected_emotions
            
        except Exception as e:
            logger.error(f"Error predicting emotions: {e}")
            return []
    
    def map_to_existing_emotions(self, detected_emotions: List[str], existing_emotions: List[str] = None) -> List[str]:
        """
        Map detected GoEmotions to existing emotion labels when possible.
        
        Args:
            detected_emotions: Emotions detected by GoEmotions
            existing_emotions: Previously known emotions
            
        Returns:
            List of mapped emotions
        """
        if not existing_emotions:
            return detected_emotions
        
        # Simple mapping - you can enhance this with more sophisticated matching
        emotion_mapping = {
            'admiration': ['respect', 'appreciation'],
            'amusement': ['humor', 'fun'],
            'anger': ['rage', 'fury', 'irritation'],
            'annoyance': ['irritation', 'frustration'],
            'approval': ['acceptance', 'agreement'],
            'caring': ['compassion', 'empathy'],
            'confusion': ['bewilderment', 'uncertainty'],
            'curiosity': ['interest', 'wonder'],
            'desire': ['longing', 'craving', 'want'],
            'disappointment': ['letdown', 'disillusionment'],
            'disapproval': ['rejection', 'disagreement'],
            'disgust': ['revulsion', 'repulsion'],
            'embarrassment': ['shame', 'awkwardness'],
            'excitement': ['enthusiasm', 'thrill'],
            'fear': ['anxiety', 'terror', 'dread'],
            'gratitude': ['thankfulness', 'appreciation'],
            'grief': ['sorrow', 'mourning'],
            'joy': ['happiness', 'delight', 'elation'],
            'love': ['affection', 'adoration'],
            'nervousness': ['anxiety', 'worry'],
            'optimism': ['hope', 'positivity'],
            'pride': ['satisfaction', 'achievement'],
            'realization': ['understanding', 'insight'],
            'relief': ['comfort', 'ease'],
            'remorse': ['regret', 'guilt'],
            'sadness': ['sorrow', 'melancholy', 'depression'],
            'surprise': ['astonishment', 'shock']
        }
        
        mapped_emotions = []
        for emotion in detected_emotions:
            # First, check if the emotion already exists
            if emotion in existing_emotions:
                mapped_emotions.append(emotion)
                continue
            
            # Try to find a match in existing emotions
            found_match = False
            if emotion in emotion_mapping:
                for mapped_emotion in emotion_mapping[emotion]:
                    if mapped_emotion in existing_emotions:
                        mapped_emotions.append(mapped_emotion)
                        found_match = True
                        break
            
            # If no match found, add the new emotion
            if not found_match:
                mapped_emotions.append(emotion)
        
        return list(set(mapped_emotions))  # Remove duplicates


# Global instance
_goemotions_detector = None

def get_goemotions_detector() -> GoEmotionsDetector:
    """Get or create the GoEmotions detector instance."""
    global _goemotions_detector
    if _goemotions_detector is None:
        _goemotions_detector = GoEmotionsDetector()
    return _goemotions_detector

async def extract_emotions_with_goemotions(
    text: str,
    existing_emotions: List[str] = None,
    threshold: float = 0.3
) -> List[str]:
    """
    Extract emotions from text using GoEmotions model.
    
    Args:
        text: Input text to analyze
        existing_emotions: Previously known emotions for mapping
        threshold: Minimum confidence threshold
        
    Returns:
        List of detected emotions
    """
    detector = get_goemotions_detector()
    detected_emotions = detector.predict_emotions(text, threshold)
    mapped_emotions = detector.map_to_existing_emotions(detected_emotions, existing_emotions)
    return mapped_emotions
