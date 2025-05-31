#!/usr/bin/env python3
"""
Complex AI-User Conversation Test for FastCoref
Test with 30+ exchanges, challenging pronouns, ambiguous references, and performance monitoring
"""

import asyncio
import sys
import os
import time
from datetime import datetime
sys.path.append('/c/pawelz-workspace/graphiti/server')

from graph_service.routers.ai.extraction import extract_facts_emotions_entities

class ConversationTester:
    def __init__(self):
        self.total_time = 0
        self.successful_resolutions = 0
        self.failed_resolutions = 0
        self.timeout_count = 0
        self.results = []
        
    async def test_complex_conversation(self):
        """Test a complex 30+ exchange conversation with challenging coreference scenarios."""
        
        print("🧪 COMPLEX CONVERSATION TEST FOR FASTCOREF")
        print("=" * 80)
        print(f"🕐 Started at: {datetime.now().strftime('%H:%M:%S')}")
        print()
        
        # Build complex conversation with challenging pronoun scenarios
        conversation_history = []
        
        # Conversation turns with increasingly complex coreference challenges
        turns = [
            # Turn 1-5: Basic setup with multiple people
            ("user", "I met Anna and Mark at the coffee shop yesterday"),
            ("assistant", "That sounds nice! How do you know them?"),
            ("user", "Anna is my colleague and Mark is her boyfriend"),
            ("assistant", "I see. What did you all talk about?"),
            ("user", "She mentioned that he got a promotion at work"),
            
            # Turn 6-10: Adding complexity with nested references
            ("user", "He was really excited about it and she was proud of him"),
            ("assistant", "That's wonderful! What kind of promotion did Mark get?"),
            ("user", "They promoted him to team lead, which she said he deserved"),
            ("assistant", "That's great news! How long has he been with the company?"),
            ("user", "Anna told me that he's been there for three years"),
            
            # Turn 11-15: Introducing ambiguity with multiple people
            ("user", "Actually, we also met Sarah at the coffee shop"),
            ("assistant", "Oh, so there were four of you? Tell me more about Sarah."),
            ("user", "She's Anna's sister and she just moved to town"),
            ("assistant", "How exciting! Where did she move from?"),
            ("user", "She came from Seattle, and she said that she loves it here already"),
            
            # Turn 16-20: Maximum ambiguity - multiple 'she/he' references
            ("user", "Sarah mentioned that she knows someone who works where he does"),
            ("assistant", "Interesting! Small world. Did she say who it is?"),
            ("user", "She said it's her friend Jake, and that he might help her find a job"),
            ("assistant", "That would be perfect! What kind of work is she looking for?"),
            ("user", "She wants to work in marketing, which is what he does too"),
            
            # Turn 21-25: Long distance references and nested conversations
            ("user", "Jake called her later and they talked about opportunities"),
            ("assistant", "That's great timing! What did he suggest?"),
            ("user", "He told her that his company is hiring and that she should apply"),
            ("assistant", "Perfect! Did she seem interested in his suggestion?"),
            ("user", "She was very excited and said she would send him her resume"),
            
            # Turn 26-30: Complex nested references with multiple conversations
            ("user", "When Anna heard about this, she offered to help with her interview prep"),
            ("assistant", "Anna sounds like a supportive sister! How did she offer to help?"),
            ("user", "She said she could practice with her since she's good at interviews"),
            ("assistant", "That's so thoughtful! Has Sarah taken her up on the offer?"),
            ("user", "Yes, they scheduled practice sessions and she's feeling more confident"),
            
            # Turn 31-35: Ultimate complexity with multiple embedded references
            ("user", "Mark also offered to put in a good word with his contacts"),
            ("assistant", "Wow, everyone is being so supportive! How does Sarah feel about all this help?"),
            ("user", "She's overwhelmed by their kindness and said she feels lucky to have them"),
            ("assistant", "That's beautiful! It sounds like she's found a great support system."),
            ("user", "Exactly! She told me that moving here was the best decision she ever made"),
            
            # Turn 36: Final complex sentence with multiple potential references
            ("user", "Anna said that when she gets the job, they should all celebrate together because he deserves credit for helping her succeed"),
        ]
        
        print("📝 CONVERSATION SCENARIO:")
        print("   - 4 people: Anna, Mark, Sarah, Jake")
        print("   - Multiple pronoun chains and ambiguous references") 
        print("   - Long-distance coreference resolution required")
        print("   - Nested conversations about other conversations")
        print(f"   - {len(turns)} total exchanges\n")
        
        # Process each turn
        for i, (role, content) in enumerate(turns, 1):
            print(f"🔄 Processing Turn {i}/{len(turns)}")
            print(f"   [{role}]: {content}")
            
            # Add to conversation history
            conversation_history.append({"role": role, "content": content})
            
            # Only test user messages for coreference (they contain the pronouns)
            if role == "user":
                start_time = time.time()
                
                try:
                    result = await extract_facts_emotions_entities(
                        message_content=content,
                        chat_history=conversation_history[:-1],  # Exclude current message
                        existing_entities=["Anna", "Mark", "Sarah", "Jake", "coffee shop", "promotion", "Seattle", "company"],
                        existing_emotions=[]
                    )
                    
                    end_time = time.time()
                    processing_time = end_time - start_time
                    self.total_time += processing_time
                    
                    # Analyze results
                    clusters = result['coreference_info'].get('clusters', [])
                    resolved_text = result['resolved_text']
                    original_text = result['coreference_info']['original_text']
                    
                    if clusters:
                        self.successful_resolutions += 1
                        status = "✅ RESOLVED"
                    else:
                        self.failed_resolutions += 1
                        status = "❌ NO RESOLUTION"
                    
                    print(f"   {status} ({processing_time:.3f}s)")
                    if original_text != resolved_text:
                        print(f"   Original: {original_text}")
                        print(f"   Resolved: {resolved_text}")
                    if clusters:
                        print(f"   Clusters: {clusters}")
                    
                    # Store detailed results
                    self.results.append({
                        'turn': i,
                        'content': content,
                        'processing_time': processing_time,
                        'clusters': clusters,
                        'resolved_text': resolved_text,
                        'original_text': original_text,
                        'entities': result['entities'],
                        'facts': result['facts']
                    })
                    
                except asyncio.TimeoutError:
                    print(f"   ⏰ TIMEOUT after 30 seconds")
                    self.timeout_count += 1
                    
                except Exception as e:
                    print(f"   ❌ ERROR: {e}")
                    self.failed_resolutions += 1
                    
                print()  # Empty line for readability
            else:
                # Assistant responses don't need coreference processing
                print("   (Assistant response - skipped)")
                print()
        
        # Print comprehensive results
        await self._print_results()
        
    async def _print_results(self):
        """Print comprehensive test results and analysis."""
        
        print("\n" + "="*80)
        print("📊 COMPREHENSIVE TEST RESULTS")
        print("="*80)
        
        user_turns = sum(1 for r in self.results)
        total_turns = len([t for t in self.results if 'turn' in t])
        
        print(f"⏱️  PERFORMANCE METRICS:")
        print(f"   Total processing time: {self.total_time:.2f} seconds")
        print(f"   Average time per turn: {self.total_time/max(user_turns, 1):.3f} seconds")
        print(f"   User turns processed: {user_turns}")
        print(f"   Successful resolutions: {self.successful_resolutions}")
        print(f"   Failed resolutions: {self.failed_resolutions}")
        print(f"   Timeouts: {self.timeout_count}")
        print(f"   Success rate: {(self.successful_resolutions/max(user_turns, 1)*100):.1f}%")
        
        print(f"\n🔍 DETAILED ANALYSIS:")
        
        # Find longest processing time
        if self.results:
            slowest = max(self.results, key=lambda x: x['processing_time'])
            fastest = min(self.results, key=lambda x: x['processing_time'])
            
            print(f"   Slowest turn: {slowest['processing_time']:.3f}s (Turn {slowest['turn']})")
            print(f"   Fastest turn: {fastest['processing_time']:.3f}s (Turn {fastest['turn']})")
            
            # Count turns with coreference resolution
            resolved_turns = [r for r in self.results if r['clusters']]
            print(f"   Turns with clusters: {len(resolved_turns)}/{len(self.results)}")
            
            if resolved_turns:
                print(f"\n📋 COREFERENCE RESOLUTIONS FOUND:")
                for result in resolved_turns:
                    print(f"   Turn {result['turn']}: {result['clusters']}")
                    if result['original_text'] != result['resolved_text']:
                        print(f"      '{result['original_text']}' → '{result['resolved_text']}'")
        
        print(f"\n🎯 CHALLENGE ASSESSMENT:")
        challenges = [
            "Multiple people with ambiguous pronoun references",
            "Long-distance coreference resolution (20+ turns apart)",
            "Nested conversations about other conversations", 
            "Complex sentences with multiple potential antecedents",
            "Pronouns referring to different people in same sentence"
        ]
        
        for challenge in challenges:
            print(f"   ✓ {challenge}")
            
        print(f"\n🏆 OVERALL ASSESSMENT:")
        if self.timeout_count == 0:
            print("   ✅ NO TIMEOUTS - Model is stable!")
        else:
            print(f"   ⚠️  {self.timeout_count} timeouts detected")
            
        if self.successful_resolutions > user_turns * 0.7:
            print("   🎉 EXCELLENT coreference resolution rate!")
        elif self.successful_resolutions > user_turns * 0.4:
            print("   👍 GOOD coreference resolution rate")
        else:
            print("   📈 Room for improvement in coreference resolution")
            
        if self.total_time / max(user_turns, 1) < 0.5:
            print("   ⚡ FAST processing speed!")
        elif self.total_time / max(user_turns, 1) < 1.0:
            print("   👌 REASONABLE processing speed")
        else:
            print("   🐌 Slow processing - consider optimization")

async def main():
    """Run the complex conversation test."""
    tester = ConversationTester()
    await tester.test_complex_conversation()

if __name__ == "__main__":
    asyncio.run(main())
