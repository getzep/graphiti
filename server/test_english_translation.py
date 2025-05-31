#!/usr/bin/env python3
"""
Test English translation of Polish example to see FastCoref behavior
Polish: "poszedłem z jarkiem do kina", "oglądaliśmy mecz tam", "powiedział że mecz był ok"
English: "I went to the cinema with Jarek", "we watched a match there", "he said the match was good"
"""

import asyncio
import sys
import os
sys.path.append('/c/pawelz-workspace/graphiti/server')

from graph_service.routers.ai.extraction import extract_facts_emotions_entities

async def test_english_translation():
    """Test the English translation of the Polish conversation example."""
    
    print("🧪 Testing English Translation of Polish Example")
    print("=" * 70)
    
    # Original Polish vs English translation
    print("📝 Original Polish Conversation:")
    polish_history = [
        "poszedłem z jarkiem do kina",
        "oglądaliśmy mecz tam", 
        "powiedział że mecz był ok"
    ]
    for i, msg in enumerate(polish_history, 1):
        print(f"   {i}. {msg}")
    
    print("\n🔄 English Translation:")
    english_history = [
        {"role": "user", "content": "I went to the cinema with Jarek"},
        {"role": "assistant", "content": "That sounds fun! How was it?"},
        {"role": "user", "content": "we watched a match there"}
    ]
    
    english_message = "he said the match was good"
    
    for i, msg in enumerate(english_history, 1):
        print(f"   {i}. [{msg['role']}]: {msg['content']}")
    print(f"   4. [user]: {english_message}")
    
    print(f"\n🔍 Key Coreference to Resolve:")
    print("   - 'he' should resolve to 'Jarek' from context")
    print("   - 'the match' should refer to match mentioned earlier")
    
    print(f"\n🚀 Running FastCoref on English Text...")
    
    try:
        result = await extract_facts_emotions_entities(
            message_content=english_message,
            chat_history=english_history,
            existing_entities=["Jarek", "cinema"],
            existing_emotions=[]
        )
        
        print(f"\n✅ RESULTS:")
        print(f"   Original Text: '{result['coreference_info']['original_text']}'")
        print(f"   Resolved Text: '{result['resolved_text']}'")
        print(f"   Entities Found: {result['entities']}")
        print(f"   Facts: {result['facts']}")
        print(f"   Emotions: {result['emotions']}")
        
        print(f"\n🔗 Coreference Information:")
        clusters = result['coreference_info'].get('clusters', [])
        if clusters:
            print(f"   Found {len(clusters)} coreference cluster(s):")
            for i, cluster in enumerate(clusters, 1):
                print(f"   Cluster {i}: {cluster}")
        else:
            print("   No coreference clusters found")
            
        print(f"\n📊 Analysis:")
        print(f"   Expected: 'Jarek said the match was good'")
        print(f"   Actual:   '{result['resolved_text']}'")
        
        # Check if coreference was resolved
        resolved_lower = result['resolved_text'].lower()
        if "jarek" in resolved_lower:
            print("   ✅ SUCCESS: 'he' was resolved to 'Jarek'!")
        elif clusters:
            print("   ⚠️  PARTIAL: Coreference detected but not fully resolved")
        else:
            print("   ❌ FAILED: No coreference resolution detected")
            
        # Compare with Polish version
        print(f"\n🆚 Comparison with Polish:")
        print("   Polish: No coreference clusters (ukryty podmiot)")
        print(f"   English: {'Coreference found!' if clusters else 'No coreference found'}")
        
        if clusters:
            print("   🎉 English translation allows FastCoref to work!")
        else:
            print("   🤔 Even English version doesn't work - may need longer context")
            
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()

async def test_direct_fastcoref():
    """Test FastCoref directly on the English text to see raw model behavior."""
    
    print(f"\n" + "="*70)
    print("🔬 DIRECT FASTCOREF TEST")
    print("="*70)
    
    from graph_service.routers.ai.coreference_resolver import get_coreference_resolver
    
    resolver = get_coreference_resolver()
    
    # Test full English context
    full_english_text = "I went to the cinema with Jarek. We watched a match there. He said the match was good."
    
    print(f"Full English Text: '{full_english_text}'")
    
    if resolver.is_available():
        try:
            result = resolver.model.predict([full_english_text])
            
            if result and len(result) > 0:
                prediction = result[0]
                clusters = prediction.get_clusters()
                
                print(f"\n📋 FastCoref Raw Results:")
                print(f"   Text: {prediction.text}")
                print(f"   Clusters found: {len(clusters)}")
                
                if clusters:
                    for i, cluster in enumerate(clusters, 1):
                        print(f"   Cluster {i}: {cluster}")
                        
                    # Try to get resolved text
                    try:
                        resolved = prediction.text
                        print(f"   Resolved: {resolved}")
                    except Exception as e:
                        print(f"   Could not get resolved text: {e}")
                else:
                    print("   ❌ No clusters found even in direct test!")
                    
            else:
                print("   ❌ No predictions returned")
                
        except Exception as e:
            print(f"   ❌ Error in direct test: {e}")
    else:
        print("   ❌ FastCoref not available")

if __name__ == "__main__":
    asyncio.run(test_english_translation())
    asyncio.run(test_direct_fastcoref())
