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
    chat_history = None  # Can be either List[Dict] with role/content or str for backward compatibility
) -> Dict[str, List[str]]:
    """
    Extract facts, emotions, and entities from message content using OpenAI function calls.
    
    Args:
        message_content: Content of the message to analyze
        existing_emotions: List of existing emotions to compare against
        existing_entities: List of existing entities to compare against
        chat_history: Chat history for context - can be either:
                     - List[Dict] with 'role' and 'content' keys (new format)
                     - str (old format, for backward compatibility)
    
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
You are an assistant tasked with extracting meaningful and concise factual statements from a user's message, using previous user messages provided in the assistant role for context clarification only.

Guidelines:
- First, check if the user's current message contains significant factual content. Ignore very short or meaningless messages such as "yes", "no", "oh", "why", "maybe", etc.
- Extract only objective, observable events or actions with significant meaning or value.
- Exclude any statements expressing thoughts, feelings, opinions, or insignificant details.
- Combine related facts into a single sentence whenever possible.
- Prioritize the user's current message. Previous user messages from the assistant context are only for resolving ambiguities about who or what the user refers to.
- If no meaningful factual statements can be extracted, return an empty list.

Example:
User message: "I love her"
Assistant previous user messages: "I spoke to Lila last night | yesterday clearly I was waiting for her to come"
Extracted Fact: ["User loves Lila"]

Output the result strictly as per the following function-calling schema:
 ["Extracted factual statement(s)"]

"""

    emotions_context = f"""
Already existing emotions: {existing_emotions or []}
You are an assistant tasked with extracting emotions from a user's message, using previously identified emotions provided for context clarification.

Guidelines:
- Extract emotions only from the user's current message.
- Match extracted emotions to existing known emotions based on meaning, synonyms, or clear similarity whenever possible.
- Return an existing emotion if a new emotion closely matches an existing one.
- Add a new emotion only if it is truly new and not already covered by known emotions.
- Be as specific as possible. Prefer detailed emotional labels (e.g., 'resentment', 'shame', 'anticipation') over general ones (e.g., 'sadness', 'happiness').
- Avoid vague or overly broad categories unless no better match exists.

Examples:

Example 1:
Known emotions: ['grief']
User message: "I still can't believe he's gone."
Extracted emotions: ['grief']

Example 2:
Known emotions: ['relief']
User message: "Finally, it's over. I can breathe again."
Extracted emotions: ['relief']

Example 3:
Known emotions: ['frustration']
User message: "I keep trying and failing. It's exhausting."
Extracted emotions: ['frustration']

Example 4:
Known emotions: ['resentment', 'envy']
User message: "She always gets what she wants. It's not fair."
Extracted emotions: ['resentment', 'envy']

Example 5:
Known emotions: ['sadness', 'anger']
User message: "Every morning I feel the same pointlessness, but I'm neither angry nor sad – just empty."
Extracted emotions: ['emptiness']

Example 6:
Known emotions: ['anxiety', 'excitement']
User message: "I can't focus because I'm constantly waiting for a reply."
Extracted emotions: ['anticipation']

Example 7:
Known emotions: ['jealousy', 'envy']
User message: "I can't be happy for their success. I feel overlooked."
Extracted emotions: ['feeling overlooked']

Example 8:
Known emotions: ['shame']
User message: "I wanted to say something smart, but I hesitated and pretended I had nothing to add."
Extracted emotions: ['self-doubt']

Output the result strictly as per the following function-calling schema:

["Extracted emotion/emotions"]
"""

    entities_context = f"""
Already existing entities: {existing_entities or []}
You are an assistant tasked with extracting entities (persons, places, objects) from a user's message, using previous known entities provided for context clarification.

Guidelines:
- Extract new entities only from the user's current message.
- When a pronoun (e.g., "he", "she", "they", "it") or vague reference (e.g., "this person") is used, always try to match it to an existing known entity based on meaning, synonyms, pronouns, grammatical gender, or clear similarity.
- If a pronoun or vague reference could match multiple known entities, return all possible matches (e.g., all female names for "she" if unclear).
- Never return a bare pronoun (such as "she", "he", etc.) as an entity. If you are about to do so, stop and make one more careful attempt to deduce the correct entity or entities using all available context.
- Add a new entity only if it is truly new and not already covered by known entities.

Examples:

Example 1:
Known entities: ['Sarah']
User message: "She helped me yesterday."
Extracted entities: ['Sarah']

Example 2:
Known entities: ['London']
User message: "I was there last summer."
Extracted entities: ['London']

Example 3:
Known entities: ['my boss']
User message: "He was very strict."
Extracted entities: ['my boss']

Example 4:
Known entities: ['David']
User message: "He called me last night."
Extracted entities: ['David']

Example 5:
Known entities: ['the old house']
User message: "I went back there to see it."
Extracted entities: ['the old house']

Example 6:
Known entities: ['Sarah', 'Anna']
User message: "She wasn't at the meeting."
Extracted entities: ['Sarah', 'Anna']

Output the result strictly as per the following function-calling schema:
 ["Extracted entity/entities"]
"""
    
    # Initialize results
    facts = []
    emotions = []
    entities = []
    
    # Prepare messages for OpenAI API calls
    base_messages = []
    
    # Add chat history to base messages if it exists
    if chat_history and isinstance(chat_history, list) and len(chat_history) > 0:
        # Convert chat_history list to OpenAI messages format
        for msg in chat_history:
            if isinstance(msg, dict) and 'role' in msg and 'content' in msg:
                # Only add non-empty messages
                if msg['content'] and msg['content'].strip():
                    base_messages.append({
                        "role": msg['role'], 
                        "content": msg['content']
                    })
    elif chat_history and isinstance(chat_history, str) and chat_history.strip():
        # Backward compatibility for string chat_history
        base_messages.append({"role": "assistant", "content": chat_history})
    
    # Add the current user message
    base_messages.append({"role": "user", "content": message_content})
    
    # 1) Extract facts
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
        emotions = json.loads(fc.arguments).get("emotions", [])    # 3) Extract entities
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
        entities = json.loads(fc.arguments).get("entities", [])
    
    # Calculate token usage
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
