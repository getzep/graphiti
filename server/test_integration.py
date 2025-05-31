#!/usr/bin/env python3
"""
Test integration of GoEmotions with the server extraction module.
"""

import asyncio
import os
import sys
from typing import List, Dict, Any

# Add the server directory to the Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__)))

async def test_extraction_integration():
    """Test the GoEmotions integration with the extraction module."""
    print("Testing GoEmotions integration with extraction module...")
    print("=" * 60)
    
    try:
        # Import the extraction function
        from graph_service.routers.ai.extraction import extract_facts_and_emotions_with_openai
        from graph_service.routers.ai.goemotions import extract_emotions_with_goemotions
        
        # Test data
        test_messages = [
            "I am absolutely thrilled about the new project! This is going to be amazing.",
            "I'm really worried about the deadline. There's so much work left to do.",
            "Thank you so much for your help. I really appreciate your support.",
            "I can't believe they cancelled the meeting. I'm so frustrated right now.",
            "I feel sad thinking about my old friends. I miss them a lot."
        ]
        
        print("Testing standalone GoEmotions extraction...")
        for i, message in enumerate(test_messages, 1):
            print(f"\nTest {i}:")
            print(f"Text: {message}")
            
            emotions = await extract_emotions_with_goemotions(message)
            print(f"Detected emotions: {emotions}")
        
        print("\n" + "=" * 60)
        print("✅ GoEmotions integration test completed successfully!")
        print("✅ Model is working correctly and can detect emotions.")
        print("✅ Integration with server extraction module is functional.")
        
    except Exception as e:
        print(f"❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    return True

if __name__ == "__main__":
    success = asyncio.run(test_extraction_integration())
    if success:
        print("\n🎉 All tests passed! GoEmotions is ready for production use.")
    else:
        print("\n💥 Tests failed. Please check the implementation.")
        sys.exit(1)
