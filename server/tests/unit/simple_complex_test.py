#!/usr/bin/env python3
"""
Simplified Complex Conversation Test
"""

import asyncio
import sys
import os
import time
from datetime import datetime
sys.path.append('/c/pawelz-workspace/graphiti/server')

from graph_service.routers.ai.extraction import extract_facts_emotions_entities

async def simple_test():
    print("🧪 SIMPLIFIED COMPLEX CONVERSATION TEST")
    print("=" * 60)
    print(f"🕐 Started at: {datetime.now().strftime('%H:%M:%S')}")
    print()
    
    # Just test a few challenging turns
    test_cases = [
        {
            'content': "I met Anna and Mark at the coffee shop yesterday",
            'history': []
        },
        {
            'content': "She mentioned that he got a promotion at work",
            'history': [
                {"role": "user", "content": "I met Anna and Mark at the coffee shop yesterday"},
                {"role": "assistant", "content": "That sounds nice! How do you know them?"},
                {"role": "user", "content": "Anna is my colleague and Mark is her boyfriend"}
            ]
        },
        {
            'content': "He was really excited about it and she was proud of him",
            'history': [
                {"role": "user", "content": "I met Anna and Mark at the coffee shop yesterday"},
                {"role": "assistant", "content": "That sounds nice! How do you know them?"},
                {"role": "user", "content": "Anna is my colleague and Mark is her boyfriend"},
                {"role": "assistant", "content": "I see. What did you all talk about?"},
                {"role": "user", "content": "She mentioned that he got a promotion at work"}
            ]
        }
    ]
    
    for i, test_case in enumerate(test_cases, 1):
        print(f"🔄 Test Case {i}: {test_case['content']}")
        
        start_time = time.time()
        try:
            result = await extract_facts_emotions_entities(
                message_content=test_case['content'],
                chat_history=test_case['history'],
                existing_entities=["Anna", "Mark", "coffee shop", "promotion"],
                existing_emotions=[]
            )
            
            end_time = time.time()
            processing_time = end_time - start_time
            
            clusters = result['coreference_info'].get('clusters', [])
            resolved_text = result['resolved_text']
            original_text = result['coreference_info']['original_text']
            
            print(f"   ✅ Success ({processing_time:.3f}s)")
            print(f"   Original: {original_text}")
            print(f"   Resolved: {resolved_text}")
            if clusters:
                print(f"   Clusters: {clusters}")
            print()
            
        except Exception as e:
            print(f"   ❌ Error: {e}")
            print()

if __name__ == "__main__":
    asyncio.run(simple_test())
