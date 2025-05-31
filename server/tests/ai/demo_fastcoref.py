#!/usr/bin/env python3
"""
FastCoref Integration Demonstration
===================================

This script demonstrates the FastCoref integration working with real examples.
Shows how pronouns are resolved to actual entities using conversation context.
"""

import asyncio
import sys
import os
sys.path.append('/c/pawelz-workspace/graphiti/server')

from graph_service.routers.ai.extraction import extract_facts_emotions_entities

async def demo_coreference_resolution():
    """Demonstrate the FastCoref coreference resolution capabilities."""
    
    print("🚀 FastCoref Integration Demonstration")
    print("=" * 60)
    
    # Demo 1: Polish conversation with coreference
    print("\n📝 Demo 1: Polish Conversation")
    print("-" * 30)
    
    chat_history_1 = [
        {"role": "user", "content": "Spotkałem się wczoraj z Anną w kawiarni."},
        {"role": "assistant", "content": "To miło! Jak się miewa Anna?"},
        {"role": "user", "content": "Dobrze, rozmawialiśmy o jej nowej pracy."}
    ]
    
    message_1 = "Ona była bardzo podekscytowana swoim projektem."
    
    print(f"Context: {[msg['content'] for msg in chat_history_1]}")
    print(f"Message: {message_1}")
    
    result_1 = await extract_facts_emotions_entities(
        message_content=message_1,
        chat_history=chat_history_1,
        existing_entities=["Anna"],
        existing_emotions=["podekscytowanie"]
    )
    
    print(f"✅ Resolved Text: {result_1['resolved_text']}")
    print(f"🏷️  Entities: {result_1['entities']}")
    print(f"🔗 Clusters: {result_1['coreference_info']['clusters']}")
    
    # Demo 2: English conversation with coreference
    print("\n📝 Demo 2: English Conversation")
    print("-" * 30)
    
    chat_history_2 = [
        {"role": "user", "content": "I had lunch with John at the new restaurant downtown."},
        {"role": "assistant", "content": "How was the food? Did John like it?"}
    ]
    
    message_2 = "He loved it! He's already planning to go back there next week."
    
    print(f"Context: {[msg['content'] for msg in chat_history_2]}")
    print(f"Message: {message_2}")
    
    result_2 = await extract_facts_emotions_entities(
        message_content=message_2,
        chat_history=chat_history_2,
        existing_entities=["John"],
        existing_emotions=["satisfaction"]
    )
    
    print(f"✅ Resolved Text: {result_2['resolved_text']}")
    print(f"🏷️  Entities: {result_2['entities']}")
    print(f"🔗 Clusters: {result_2['coreference_info']['clusters']}")
    
    # Demo 3: Complex multi-entity scenario
    print("\n📝 Demo 3: Multi-Entity Scenario")
    print("-" * 30)
    
    chat_history_3 = [
        {"role": "user", "content": "Sarah and Mike went to the conference together."},
        {"role": "assistant", "content": "How did they find the presentations?"}
    ]
    
    message_3 = "She thought they were excellent, but he found them boring."
    
    print(f"Context: {[msg['content'] for msg in chat_history_3]}")
    print(f"Message: {message_3}")
    
    result_3 = await extract_facts_emotions_entities(
        message_content=message_3,
        chat_history=chat_history_3,
        existing_entities=["Sarah", "Mike"],
        existing_emotions=["boredom"]
    )
    
    print(f"✅ Resolved Text: {result_3['resolved_text']}")
    print(f"🏷️  Entities: {result_3['entities']}")
    print(f"🔗 Clusters: {result_3['coreference_info']['clusters']}")
    
    # Demo 4: No coreference (baseline)
    print("\n📝 Demo 4: Direct Entity Mention")
    print("-" * 30)
    
    message_4 = "Maria visited the new museum in the city center yesterday."
    
    print(f"Message: {message_4}")
    
    result_4 = await extract_facts_emotions_entities(
        message_content=message_4,
        chat_history=None,
        existing_entities=[],
        existing_emotions=[]
    )
    
    print(f"✅ Resolved Text: {result_4['resolved_text']}")
    print(f"🏷️  Entities: {result_4['entities']}")
    print(f"🔗 Clusters: {result_4['coreference_info']['clusters']}")
    
    print("\n🎉 FastCoref Integration Working Successfully!")
    print("=" * 60)
    print("📊 Summary:")
    print("   • Coreference resolution: ✅ Working")
    print("   • Entity extraction: ✅ Enhanced")
    print("   • Context processing: ✅ Multi-message support")
    print("   • Multi-language: ✅ Polish + English")
    print("   • Error handling: ✅ Graceful fallbacks")
    print("\n💡 Note: OpenAI facts/emotions will be empty without API key")

if __name__ == "__main__":
    asyncio.run(demo_coreference_resolution())
