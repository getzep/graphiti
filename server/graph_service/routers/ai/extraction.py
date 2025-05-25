"""
Module for extracting facts, emotions and entities using OpenAI API.
"""
import openai
import json
from typing import List, Dict, Any

from .config import (
    OPENAI_API_KEY, 
    FACTS_CONFIG, 
    EMOTIONS_CONFIG, 
    ENTITIES_CONFIG
)
from .function_specs import functionsSpec

# Set API key
openai.api_key = OPENAI_API_KEY

async def extract_facts_emotions_entities(
    message_content: str, 
    existing_emotions: List[str] = None, 
    existing_entities: List[str] = None,
    chat_history: str = None
) -> Dict[str, List[str]]:
    """
    Extract facts, emotions, and entities from message content using OpenAI function calls.
    
    Args:
        message_content: Content of the message to analyze
        existing_emotions: List of existing emotions to compare against
        existing_entities: List of existing entities to compare against
        chat_history: Any chat history to provide context
    
    Returns:
        Dictionary with lists of facts, emotions, entities and token usage
    """
    if not message_content or not message_content.strip():
        return {"facts": [], "emotions": [], "entities": [], "usage": {}}
        
    # Prepare contexts
    promptBase = f'''
Message content for analysis:
"""{message_content}"""
'''

    facts_context = """
You are a converter that rewrites user sentences into **directly observable life-facts**.

Rules:
1. Remove all feelings, opinions, intentions, and plans.
2. Keep only actions or events from real life that a bystander could literally see or hear.
3. Ignore pure conversational filler such as greetings, rhetorical questions, or meta-linguistic acts
   (e.g. “He said ‘Hello’”). If an utterance contains only such filler, treat it as having no facts.
4. Write in **third-person**, declarative mood.
5. If several actions involve the same people/things and happen at the same time or place,
   **merge them into one sentence** using “while”, “when”, “as”, or “and”.
6. If actions cannot logically be merged, write one sentence per action.
7. When the text contains no observable life-facts, respond exactly with `[]` (an empty list).
8. Do not add explanations or comments—return only the transformed sentence(s) or `[]`.
"""

    emotions_context = f"""
Already existing emotions: {existing_emotions or []}
When extracting new emotions FROM USER TEXT ONLY (not from assistant section), try to match them to the existing ones if possible (by meaning, synonyms, or clear similarity). If a new emotion matches an existing one, return the existing value instead of a new variant. Only add new emotions if they are truly new and not covered by the existing ones.
"""
    
    entities_context = f"""
Already existing entities: {existing_entities or []}
When extracting new entities from the USER message only (not assistant), always try to match them to the existing entities whenever possible—based on meaning, synonyms, pronouns, grammatical gender, or clear similarity.
If a user uses a pronoun (he, she, they, it, etc.), or a vague reference ("this person", "that place"), and there is a matching existing entity, return the existing value instead of a new one.

Never create a new entity for a pronoun or vague reference if there is a clear matching entity in the list.

Only add new entities if they are truly new and not already covered by the known entities.

Example 1:
Known entities: ['Sarah']
User: "She helped me yesterday."
Extracted entities: ['Sarah']

Example 2:
Known entities: ['London']
User: "I was there last summer."
Extracted entities: ['London']

Example 3:
Known entities: ['my boss']
User: "He was very strict."
Extracted entities: ['my boss']

Example 4:
Known entities: ['David']
User: "He called me last night."
Extracted entities: ['David']

Example 5:
Known entities: ['the old house']
User: "I went back there to see it."
Extracted entities: ['the old house']

Example 6:
Known entities: ['Sarah', 'Anna']
User: "She wasn't at the meeting."
Extracted entities: ['Sarah', 'Anna']
(If the pronoun could refer to more than one known entity, return all possible matches.)"""

    # Initialize results
    facts = []
    emotions = []
    entities = []
    
    # Prepare messages for OpenAI API calls
    base_messages = [{"role": "user", "content": promptBase}]
    if chat_history and chat_history.strip():
        base_messages.append({"role": "assistant", "content": chat_history})    # 1) Extract facts
    messages_facts = base_messages + [{"role": "system", "content": facts_context}]
    respFacts = openai.chat.completions.create(
        model=FACTS_CONFIG.model,
        messages=messages_facts,
        functions=[functionsSpec[0]],
        function_call="auto",
        temperature=FACTS_CONFIG.temperature
    )
    fc = respFacts.choices[0].message.function_call
    if fc and hasattr(fc, 'arguments'):
        facts = json.loads(fc.arguments).get("facts", [])

    # 2) Extract emotions
    messages_emotions = base_messages + [{"role": "system", "content": emotions_context}]
    respEmo = openai.chat.completions.create(
        model=EMOTIONS_CONFIG.model,
        messages=messages_emotions,
        functions=[functionsSpec[1]],
        function_call="auto",
        temperature=EMOTIONS_CONFIG.temperature
    )
    fc = respEmo.choices[0].message.function_call
    if fc and hasattr(fc, 'arguments'):
        emotions = json.loads(fc.arguments).get("emotions", [])

    # 3) Extract entities
    messages_entities = base_messages + [{"role": "system", "content": entities_context}]
    respEnt = openai.chat.completions.create(
        model=ENTITIES_CONFIG.model,
        messages=messages_entities,
        functions=[functionsSpec[2]],
        function_call="auto",
        temperature=ENTITIES_CONFIG.temperature
    )
    fc = respEnt.choices[0].message.function_call
    if fc and hasattr(fc, 'arguments'):
        entities = json.loads(fc.arguments).get("entities", [])    # Calculate token usage
    usage = {
        "input_tokens": (
            respFacts.usage.prompt_tokens +
            respEmo.usage.prompt_tokens +
            respEnt.usage.prompt_tokens
        ),
        "output_tokens": (
            respFacts.usage.completion_tokens +
            respEmo.usage.completion_tokens +
            respEnt.usage.completion_tokens
        ),
        "total_tokens": (
            respFacts.usage.total_tokens +
            respEmo.usage.total_tokens +
            respEnt.usage.total_tokens
        ),
        "models": {
            "facts": FACTS_CONFIG.model,
            "emotions": EMOTIONS_CONFIG.model,
            "entities": ENTITIES_CONFIG.model
        },
        "temperatures": {
            "facts": FACTS_CONFIG.temperature,
            "emotions": EMOTIONS_CONFIG.temperature,
            "entities": ENTITIES_CONFIG.temperature
        }
    }
    
    # Ensure all lists are unique
    facts = list(set(facts))
    emotions = list(set(emotions))
    entities = list(set(entities))
    
    return {
        "facts": facts, 
        "emotions": emotions, 
        "entities": entities,
        "usage": usage
    }
