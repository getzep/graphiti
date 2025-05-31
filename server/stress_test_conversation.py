#!/usr/bin/env python3
"""
STRESS TEST: Long AI-User Conversation with Complex Coreference Challenges
==========================================================================

This test creates a 30+ exchange conversation with the most challenging 
coreference resolution scenarios for FastCoref:
- Multiple characters with similar pronouns
- Ambiguous pronoun references  
- Long conversation context
- Nested conversations about other conversations
- Complex temporal references
- Mixed languages
- Technical jargon with pronouns

GOALS:
1. Test FastCoref performance with long context
2. Check for model hanging with complex scenarios
3. Measure processing time for each message
4. Identify the most challenging coreference cases
"""

import sys
import os
import time
import asyncio

# Add the server module to Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

try:
    from graph_service.routers.ai.extraction import extract_facts_emotions_entities
    EXTRACTION_AVAILABLE = True
    print("✅ Successfully imported extraction module")
except ImportError as e:
    print(f"⚠️  Extraction module not available: {e}")
    print("Running in demo mode...")
    EXTRACTION_AVAILABLE = False
    
import logging

# Set up detailed logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def create_complex_conversation():
    """
    Create a complex 30+ exchange conversation designed to challenge FastCoref.
    Includes multiple difficult coreference scenarios.
    """
    
    conversation = [
        # Opening - introduce multiple characters
        {
            "speaker": "user",
            "text": "I met Anna and Mark at the coffee shop yesterday. She was wearing a red dress.",
            "challenge": "Simple he/she with two potential antecedents"
        },
        {
            "speaker": "ai", 
            "text": "That sounds nice! Did you enjoy talking with them?",
            "challenge": "Plural pronoun 'them' referring to both"
        },
        {
            "speaker": "user",
            "text": "Yes, but he seemed a bit distracted. Anna mentioned that his project at work is stressing him out.",
            "challenge": "Multiple pronouns: he, his, him - all should refer to Mark"
        },
        {
            "speaker": "ai",
            "text": "Work stress can be difficult. What kind of project is he working on?",
            "challenge": "Distant reference to Mark through context"
        },
        {
            "speaker": "user", 
            "text": "It's a machine learning system for their company. They want him to finish it by next month.",
            "challenge": "Nested pronouns: 'it' (project), 'their' (company), 'him' (Mark)"
        },
        {
            "speaker": "ai",
            "text": "That's a tight deadline! Has she offered to help him with it?",
            "challenge": "Ambiguous 'she' - could be Anna or someone else"
        },
        {
            "speaker": "user",
            "text": "Anna? Yes, she's actually quite good at programming. She offered, but he's too proud to accept help.",
            "challenge": "Clarification then multiple pronouns for both characters"
        },
        {
            "speaker": "ai",
            "text": "Pride can get in the way sometimes. Maybe you could suggest they work together as equals?",
            "challenge": "Plural 'they' referring to Anna and Mark working together"
        },
        {
            "speaker": "user",
            "text": "Good idea! Actually, I should mention - when we were talking, Sarah joined us. She knows both of them from university.",
            "challenge": "NEW CHARACTER INTRODUCTION - now 'she' could refer to Anna or Sarah"
        },
        {
            "speaker": "ai",
            "text": "Oh, so now there were four of you? That must have made the conversation more interesting.",
            "challenge": "Quantification reference to all people mentioned"
        },
        
        # Mid-conversation - increase complexity
        {
            "speaker": "user",
            "text": "Exactly! Sarah mentioned that when she and Anna were students, they worked on a similar project. She said it took them months to complete.",
            "challenge": "VERY COMPLEX: multiple 'she/they' in nested past context"
        },
        {
            "speaker": "ai",
            "text": "That experience could be valuable for Mark's current project. Did she share any specific insights about it?",
            "challenge": "Ambiguous 'she' (Sarah or Anna?) and 'it' (which project?)"
        },
        {
            "speaker": "user",
            "text": "Sarah did. She said the biggest mistake they made was not testing it early enough. Anna agreed and said she learned that lesson the hard way.",
            "challenge": "Multiple speakers, nested quotes, pronoun attribution"
        },
        {
            "speaker": "ai",
            "text": "Testing is crucial in software development. When Anna said she learned it the hard way, what did she mean?",
            "challenge": "Reported speech with embedded pronouns"
        },
        {
            "speaker": "user",
            "text": "Well, in their university project, Anna and Sarah spent weeks coding without testing. When they finally tested it, everything broke. She was devastated.",
            "challenge": "EXTREMELY AMBIGUOUS: 'she' could be Anna or Sarah"
        },
        {
            "speaker": "ai",
            "text": "That sounds frustrating! I imagine both of them learned from that experience.",
            "challenge": "Safe plural reference to avoid ambiguity"
        },
        {
            "speaker": "user",
            "text": "Definitely. But here's the interesting part - Mark was actually their teaching assistant back then! He helped them debug it.",
            "challenge": "Temporal shift: Mark now connected to past context"
        },
        {
            "speaker": "ai",
            "text": "What a small world! So he already knows their working style and the challenges they faced with similar projects.",
            "challenge": "Multiple pronoun references across temporal contexts"
        },
        {
            "speaker": "user",
            "text": "Exactly! That's why Sarah suggested that he should definitely accept Anna's help. She thinks his pride is silly given their history.",
            "challenge": "Nested attribution: Sarah's opinion about Mark and Anna"
        },
        {
            "speaker": "ai",
            "text": "Sarah makes a good point. Given that he helped them before, it would be natural for her to help him now.",
            "challenge": "Reciprocal relationship pronouns"
        },
        
        # Peak complexity - conversation about conversation
        {
            "speaker": "user",
            "text": "Right! So I suggested to Mark that he should call Anna tonight. But then Sarah said she already texted her about it.",
            "challenge": "NESTED COMMUNICATION: who texted whom about what?"
        },
        {
            "speaker": "ai",
            "text": "So Sarah texted Anna about Mark's project? That's taking initiative! How did she react to receiving that message?",
            "challenge": "Clarification attempt + pronoun for message recipient"
        },
        {
            "speaker": "user",
            "text": "Anna was happy about it! She immediately called Mark and said she'd be over tomorrow to help. He was surprised but agreed.",
            "challenge": "Chain reaction: Sarah→Anna→Mark with pronouns throughout"
        },
        {
            "speaker": "ai",
            "text": "That worked out perfectly! It sounds like she's a good friend for taking charge like that.",
            "challenge": "Ambiguous 'she' - could praise Sarah or Anna"
        },
        {
            "speaker": "user",
            "text": "Sarah really is! She has this way of making things happen. Anna appreciates it, and Mark benefits from it even when he's too stubborn to ask.",
            "challenge": "Multiple 'it' references to different concepts"
        },
        {
            "speaker": "ai",
            "text": "Some people are natural facilitators. It sounds like she brings out the best in her friends.",
            "challenge": "Generic 'she' and 'her friends' - scope ambiguity"
        },
        {
            "speaker": "user",
            "text": "Absolutely! Actually, this reminds me of another time when she helped someone. My brother Jake was struggling with his thesis, and she connected him with a professor she knew.",
            "challenge": "NEW CHARACTERS: Jake, unnamed professor + complex pronoun web"
        },
        {
            "speaker": "ai",
            "text": "She sounds like a wonderful person to have in your network! Did her introduction help him finish his thesis?",
            "challenge": "Distant reference chains: Sarah→professor→Jake→thesis"
        },
        {
            "speaker": "user",
            "text": "It did! Jake met with the professor, and he gave him some excellent advice. He was able to restructure it completely and defend it successfully.",
            "challenge": "AMBIGUOUS PRONOUN STORM: multiple 'he/him' could be Jake or professor"
        },
        {
            "speaker": "ai",
            "text": "That's fantastic! Academic guidance can make such a difference. I'm glad he was receptive to the professor's suggestions.",
            "challenge": "Attempt to resolve ambiguity with context"
        },
        {
            "speaker": "user",
            "text": "Yes, Jake was very grateful. He sent Sarah flowers and wrote her a long thank-you note. She was touched by his gesture.",
            "challenge": "Clear resolution then new pronoun chain"
        },
        
        # Final complexity - multilingual and technical
        {
            "speaker": "ai",
            "text": "That's so sweet! It shows how much her help meant to him.",
            "challenge": "Simple resolution to test if model maintains context"
        },
        {
            "speaker": "user",
            "text": "Definitely! You know, speaking of help, my colleague Maria asked me yesterday: '¿Crees que ella podría ayudarme también?' She was talking about Sarah.",
            "challenge": "MULTILINGUAL: Spanish embedded with English pronouns"
        },
        {
            "speaker": "ai",
            "text": "Interesting! So Maria was asking if she thought Sarah could help her too? What kind of help was she looking for?",
            "challenge": "Translation handling + multiple 'she/her' disambiguation"
        },
        {
            "speaker": "user",
            "text": "Maria is working on a neural network architecture, and she heard about Mark's project through the grapevine. She thought maybe Sarah could introduce her to Anna, since she's good with ML.",
            "challenge": "TECHNICAL + SOCIAL: complex network of referrals and expertise"
        },
        {
            "speaker": "ai",
            "text": "That's a smart networking strategy! It sounds like Sarah has become the hub connecting everyone who needs technical help.",
            "challenge": "Abstract reference to Sarah's role in the network"
        },
        {
            "speaker": "user",
            "text": "Exactly! It's funny how one coffee shop conversation led to all this. Mark gets help with his project, Jake finished his thesis, and now Maria might get connected too. Sarah started it all.",
            "challenge": "SUMMARY WITH EXTENSIVE BACKWARDS REFERENCES"
        },
        {
            "speaker": "ai",
            "text": "It's amazing how one person's initiative can create such a positive ripple effect. She really has a gift for bringing people together.",
            "challenge": "Final abstract reference to Sarah's impact"
        }
    ]
    
    return conversation

async def run_async_stress_test():
    """Run the async stress test using the extraction module."""
    
    conversation = create_complex_conversation()
    
    # Tracking variables
    total_processing_time = 0
    successful_resolutions = 0
    failed_resolutions = 0
    timeouts = 0
    context_history = []
    
    print(f"\n🎬 Starting conversation processing...")
    
    for i, turn in enumerate(conversation, 1):
        print(f"\n--- Turn {i}/36 ---")
        print(f"👤 {turn['speaker'].upper()}: {turn['text']}")
        print(f"🎯 Challenge: {turn['challenge']}")
        
        # Only process user messages
        if turn['speaker'] == 'user':
            start_time = time.time()
            
            try:
                result = await asyncio.wait_for(
                    extract_facts_emotions_entities(
                        message_content=turn['text'],
                        chat_history=[{"role": "user", "content": msg} for msg in context_history[-10:]],
                        existing_entities=["Anna", "Mark", "Sarah", "Jake", "Maria", "Professor"],
                        existing_emotions=[]
                    ),
                    timeout=30.0
                )
                
                processing_time = time.time() - start_time
                total_processing_time += processing_time
                
                # Check results
                coreference_info = result.get('coreference_info', {})
                clusters = coreference_info.get('clusters', [])
                resolved_text = result.get('resolved_text', turn['text'])
                original_text = coreference_info.get('original_text', turn['text'])
                entities = result.get('entities', [])
                
                print(f"⏱️  Processing time: {processing_time:.3f}s")
                print(f"🔗 Clusters found: {len(clusters)}")
                
                if len(clusters) > 0 or original_text != resolved_text:
                    successful_resolutions += 1
                    print(f"✅ Coreference detected!")
                    if clusters:
                        for j, cluster in enumerate(clusters, 1):
                            print(f"   Cluster {j}: {cluster}")
                    
                    if resolved_text != turn['text']:
                        print(f"🔄 Original: {original_text}")
                        print(f"🔄 Resolved: {resolved_text}")
                else:
                    failed_resolutions += 1
                    print(f"❌ No coreference found")
                
                print(f"👥 Entities: {entities}")
                
            except asyncio.TimeoutError:
                processing_time = time.time() - start_time
                total_processing_time += processing_time
                timeouts += 1
                failed_resolutions += 1
                print(f"⏰ TIMEOUT after {processing_time:.3f}s")
                
            except Exception as e:
                processing_time = time.time() - start_time
                total_processing_time += processing_time
                failed_resolutions += 1
                print(f"💥 ERROR after {processing_time:.3f}s: {e}")
        
        # Add to context for next turn
        context_history.append(turn['text'])
        
        # Short pause to prevent overwhelming
        await asyncio.sleep(0.1)
    
    # Print final statistics
    print_final_stats(total_processing_time, successful_resolutions, failed_resolutions, timeouts)

def print_final_stats(total_time, successful, failed, timeouts):
    """Print comprehensive final statistics."""
    total_turns = 36
    user_turns = len([t for t in create_complex_conversation() if t['speaker'] == 'user'])
    
    print(f"\n" + "=" * 80)
    print(f"📊 STRESS TEST RESULTS")
    print(f"=" * 80)
    print(f"🎯 Performance Stats:")
    print(f"   Total processing time: {total_time:.2f}s")
    print(f"   Average time per turn: {total_time/max(user_turns, 1):.3f}s")
    print(f"   User turns processed: {user_turns}")
    print(f"   Successful resolutions: {successful}/{user_turns} ({successful/max(user_turns, 1)*100:.1f}%)")
    print(f"   Failed resolutions: {failed}/{user_turns}")
    print(f"   Timeouts: {timeouts}/{user_turns}")
    
    print(f"\n🎯 Stability Assessment:")
    if timeouts == 0:
        print(f"   ✅ NO TIMEOUTS - Model is stable!")
    else:
        print(f"   ⚠️  {timeouts} timeouts detected - may need optimization")
    
    if total_time < 60:
        print(f"   ✅ FAST PROCESSING - Under 1 minute total")
    elif total_time < 180:
        print(f"   ⚠️  MODERATE SPEED - {total_time/60:.1f} minutes total")
    else:
        print(f"   ❌ SLOW PROCESSING - {total_time/60:.1f} minutes total")
    
    print(f"\n🎯 Challenge Assessment:")
    if successful > user_turns * 0.7:
        print(f"   ✅ EXCELLENT - Handles complex coreference well")
    elif successful > user_turns * 0.5:
        print(f"   ✅ GOOD - Handles most coreference cases")
    elif successful > user_turns * 0.3:
        print(f"   ⚠️  MODERATE - Struggles with complex cases")
    else:
        print(f"   ❌ POOR - Needs improvement for complex scenarios")
    
    print(f"\n🏆 CONCLUSION:")
    if timeouts == 0 and successful > user_turns * 0.6 and total_time < 120:
        print(f"   🎉 STRESS TEST PASSED! FastCoref is production-ready.")
    else:
        print(f"   🔧 OPTIMIZATION NEEDED - Some issues detected.")
    
    print(f"=" * 80)

def run_stress_test():
    """Run the comprehensive stress test with timing and error monitoring."""
    
    print("🚀 FASTCOREF STRESS TEST - Complex Conversation")
    print("=" * 80)
    print(f"📊 Test Parameters:")
    print(f"   - Conversation length: 30+ exchanges")
    print(f"   - Multiple characters: Anna, Mark, Sarah, Jake, Maria, Professor")
    print(f"   - Challenge types: Ambiguous pronouns, nested contexts, multilingual")
    print(f"   - Performance monitoring: Timing + error detection")
    print(f"   - Stability test: No hanging allowed")
    print("=" * 80)
    
    if not EXTRACTION_AVAILABLE:
        print("❌ Extraction module not available - cannot run stress test")
        print("Please check your Python path and module imports")
        return
    
    # Run async stress test
    asyncio.run(run_async_stress_test())
      # Tracking variables
    total_processing_time = 0
    successful_resolutions = 0
    failed_resolutions = 0
    timeouts = 0
    context_history = []
    
    print(f"\n🎬 Starting conversation processing...")
    
    # Print final statistics
    print_final_stats(total_processing_time, successful_resolutions, failed_resolutions, timeouts)

if __name__ == "__main__":
    run_stress_test()
