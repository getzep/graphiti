#!/usr/bin/env python3

import sys
import os
sys.path.append('/c/pawelz-workspace/graphiti/server')

# Test the FastCoref integration without circular imports
from fastcoref import FCoref
import spacy
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_coreference_resolution():
    """Test the coreference resolution functionality."""
    
    print("=== Testing FastCoref Coreference Resolution ===")
    
    # Initialize FastCoref
    model = FCoref(device='cpu')
    
    # Load spaCy model
    try:
        nlp = spacy.load("en_core_web_sm")
        print("✓ spaCy model loaded")
    except OSError:
        print("✗ spaCy model not found")
        return
    
    # Test cases
    test_cases = [
        {
            "context": ["I met Sarah at the coffee shop yesterday."],
            "text": "She told me about her new job.",
            "expected_entities": ["Sarah"]
        },
        {
            "context": ["Spotkałem się wczoraj z Jankiem.", "To miło! Jak się miewa Janek?"],
            "text": "On opowiadał mi o swoim nowym projekcie.",
            "expected_entities": ["Janek"]
        },
        {
            "context": [],
            "text": "John went to the store. He bought milk and eggs.",
            "expected_entities": ["John"]
        }
    ]
    
    for i, test_case in enumerate(test_cases, 1):
        print(f"\n--- Test Case {i} ---")
        context = test_case["context"]
        text = test_case["text"]
        
        # Prepare full context
        if context:
            full_context = " ".join(context) + " " + text
        else:
            full_context = text
            
        print(f"Context: {context}")
        print(f"Text: {text}")
        print(f"Full context: {full_context}")
        
        # Run FastCoref
        try:
            predictions = model.predict(texts=[full_context])
            
            if predictions and len(predictions) > 0:
                result = predictions[0]
                resolved_text = getattr(result, 'text', full_context)
                clusters = getattr(result, 'clusters', [])
                
                print(f"Resolved text: {resolved_text}")
                print(f"Clusters: {clusters}")
                
                # Extract entities from clusters
                entities = []
                for cluster in clusters:
                    # Find best mention in cluster (non-pronoun, longest)
                    best_mention = None
                    best_score = 0
                    
                    for mention in cluster:
                        mention_text = mention.strip() if isinstance(mention, str) else str(mention).strip()
                        
                        # Score mentions
                        score = len(mention_text.split())
                        if not is_pronoun(mention_text):
                            score += 10
                        if mention_text and mention_text[0].isupper():
                            score += 5
                            
                        if score > best_score:
                            best_score = score
                            best_mention = mention_text
                    
                    if best_mention and not is_pronoun(best_mention):
                        entities.append(best_mention)
                
                # Also extract with spaCy
                doc = nlp(resolved_text)
                for ent in doc.ents:
                    if ent.label_ in ["PERSON", "ORG", "GPE", "LOC"]:
                        entities.append(ent.text.strip())
                
                entities = list(set(entities))
                print(f"Extracted entities: {entities}")
                
                # Check if expected entities are found
                expected = test_case["expected_entities"]
                found_expected = any(exp.lower() in [e.lower() for e in entities] for exp in expected)
                
                if found_expected:
                    print("✓ Expected entities found")
                else:
                    print(f"✗ Expected entities {expected} not found in {entities}")
                    
            else:
                print("✗ No predictions returned")
                
        except Exception as e:
            print(f"✗ Error in coreference resolution: {e}")

def is_pronoun(text):
    """Check if text is a pronoun."""
    if not text:
        return False
        
    pronouns = {
        # English pronouns
        'he', 'she', 'it', 'they', 'him', 'her', 'them', 'his', 'hers', 'its', 'their',
        'i', 'you', 'we', 'me', 'us', 'my', 'your', 'our',
        # Polish pronouns
        'on', 'ona', 'ono', 'oni', 'one', 'go', 'ją', 'je', 'ich', 'im', 'nimi',
        'ja', 'ty', 'my', 'wy', 'mnie', 'cię', 'nas', 'was', 'mój', 'twój', 'nasz', 'wasz'
    }
    return text.lower().strip() in pronouns

if __name__ == "__main__":
    test_coreference_resolution()
