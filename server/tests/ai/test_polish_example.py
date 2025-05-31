#!/usr/bin/env python3
"""
Test Polish example: "poszedłem z jarkiem do kina", "oglądaliśmy mecz tam", "powiedział że mecz był ok"
"""

import asyncio
import sys
import os
sys.path.append('/c/pawelz-workspace/graphiti/server')

from graph_service.routers.ai.extraction import extract_facts_emotions_entities

async def test_polish_example():
    """Test the specific Polish conversation example."""
    
    print("🧪 Testing Polish Coreference Example")
    print("=" * 60)
    
    # Your conversation sequence
    chat_history = [
        {"role": "user", "content": "poszedłem z jarkiem do kina"},
        {"role": "assistant", "content": "To brzmi fajnie! Jak było?"},
        {"role": "user", "content": "oglądaliśmy mecz tam"}
    ]
    
    current_message = "powiedział że mecz był ok"
    
    print(f"📝 Chat History:")
    for i, msg in enumerate(chat_history, 1):
        print(f"   {i}. [{msg['role']}]: {msg['content']}")
    
    print(f"\n💬 Current Message: '{current_message}'")
    print(f"\n🔍 Analysis:")
    print("   - 'powiedział' = he said")
    print("   - Should resolve 'he' to 'Jarek' from context")
    print("   - 'mecz' should refer to the match mentioned earlier")
    print("   - 'tam' in previous message refers to 'kino' (cinema)")
    
    print(f"\n🚀 Running FastCoref...")
    
    try:
        result = await extract_facts_emotions_entities(
            message_content=current_message,
            chat_history=chat_history,
            existing_entities=["Jarek", "kino"],
            existing_emotions=[]
        )
        
        print(f"\n✅ Results:")
        print(f"   Original Text: '{result['coreference_info']['original_text']}'")
        print(f"   Resolved Text: '{result['resolved_text']}'")
        print(f"   Entities Found: {result['entities']}")
        print(f"   Facts: {result['facts']}")
        print(f"   Emotions: {result['emotions']}")
        
        print(f"\n🔗 Coreference Information:")
        clusters = result['coreference_info'].get('clusters', [])
        if clusters:
            for i, cluster in enumerate(clusters):
                print(f"   Cluster {i+1}: {cluster}")
        else:
            print("   No coreference clusters found")
            
        print(f"\n🤔 Expected vs Actual:")
        print(f"   Expected: 'Jarek powiedział że mecz był ok' (Jarek said the match was ok)")
        print(f"   Actual:   '{result['resolved_text']}'")
        
        if "jarek" in result['resolved_text'].lower() or "Jarek" in result['resolved_text']:
            print("   ✅ Successfully resolved 'powiedział' to 'Jarek'!")
        else:
            print("   ❌ Did not resolve 'powiedział' to 'Jarek'")
            
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(test_polish_example())
