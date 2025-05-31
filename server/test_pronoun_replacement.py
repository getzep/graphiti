#!/usr/bin/env python3
"""
Test actual pronoun replacement using FastCoref clusters.
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), 'graph_service', 'routers', 'ai'))

try:
    from coreference_resolver import CoreferenceResolver
    COREFERENCE_RESOLVER_AVAILABLE = True
except ImportError as e:
    print(f"⚠️  CoreferenceResolver not available: {e}")
    COREFERENCE_RESOLVER_AVAILABLE = False
    
from fastcoref import FCoref
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def replace_pronouns_with_entities(text: str, clusters):
    """
    Replace pronouns with their resolved entities based on coreference clusters.
    
    Args:
        text: Original text
        clusters: Coreference clusters from FastCoref
        
    Returns:
        Text with pronouns replaced
    """
    result_text = text
    
    for cluster in clusters:
        if len(cluster) < 2:
            continue
            
        # Find the best entity (usually the first full name/proper noun)
        entity = None
        pronouns = []
        
        for mention in cluster:
            mention_lower = mention.lower()
            if mention_lower in ['he', 'she', 'it', 'they', 'him', 'her', 'them', 'his', 'her', 'hers', 'its', 'their', 'theirs']:
                pronouns.append(mention)
            else:
                # This looks like a proper entity
                if entity is None or len(mention) > len(entity):
                    entity = mention
        
        # Replace pronouns with the entity
        if entity and pronouns:
            for pronoun in pronouns:
                # Case-sensitive replacement
                result_text = result_text.replace(f" {pronoun} ", f" {entity} ")
                result_text = result_text.replace(f" {pronoun}.", f" {entity}.")
                result_text = result_text.replace(f" {pronoun},", f" {entity},")
                
                # Handle start of sentence
                if result_text.startswith(f"{pronoun} "):
                    result_text = result_text.replace(f"{pronoun} ", f"{entity} ", 1)
    
    return result_text

def test_manual_pronoun_replacement():
    """Test manual pronoun replacement using FastCoref clusters."""
    
    print("🔧 Testing Manual Pronoun Replacement")
    print("=" * 60)
    
    # English translation example
    english_text = "I went to the cinema with Jarek. We watched a match there. He said the match was good."
    
    print(f"📝 Original Text: '{english_text}'")
    
    # Get FastCoref results
    model = FCoref(device='cpu')
    predictions = model.predict(
        texts=[english_text],
        output_file=None,
        is_split_into_words=False
    )
    
    if predictions and len(predictions) > 0:
        result = predictions[0]
        clusters = result.get_clusters()
        
        print(f"🔗 Found {len(clusters)} coreference cluster(s):")
        for i, cluster in enumerate(clusters, 1):
            print(f"   Cluster {i}: {cluster}")
        
        # Apply manual pronoun replacement
        resolved_text = replace_pronouns_with_entities(english_text, clusters)
        
        print(f"✅ Resolved Text: '{resolved_text}'")
        
        # Check if replacement worked
        if "Jarek said" in resolved_text:
            print("🎉 SUCCESS: 'He' was replaced with 'Jarek'!")
        else:
            print("⚠️  PARTIAL: Replacement didn't work as expected")
    else:
        print("❌ No predictions from FastCoref")

def test_with_coreference_resolver():
    """Test using our enhanced CoreferenceResolver."""
    
    print("\n🔧 Testing with CoreferenceResolver")
    print("=" * 60)
    
    if not COREFERENCE_RESOLVER_AVAILABLE:
        print("❌ CoreferenceResolver not available, skipping test")
        return
    
    resolver = CoreferenceResolver()
    
    # Test with conversation context
    context_history = [
        "I went to the cinema with Jarek.",
        "We watched a match there."
    ]
    current_message = "He said the match was good."
    
    print(f"📝 Context: {context_history}")
    print(f"📝 Current: '{current_message}'")
    
    result = resolver.resolve_coreferences_and_extract_entities(
        text=current_message,
        context_history=context_history
    )
    
    print(f"✅ Results:")
    print(f"   Resolved Text: '{result['resolved_text']}'")
    print(f"   Entities: {result['entities']}")
    print(f"   Clusters: {len(result['coreference_clusters'])}")
    
    # Check if our enhanced resolver works better
    if "Jarek" in result['resolved_text']:
        print("🎉 SUCCESS: CoreferenceResolver resolved the pronoun!")
    else:
        print("⚠️  PARTIAL: CoreferenceResolver didn't fully resolve")

if __name__ == "__main__":
    test_manual_pronoun_replacement()
    test_with_coreference_resolver()
