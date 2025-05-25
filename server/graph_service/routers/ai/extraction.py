"""
Module for extracting facts, emotions and entities using OpenAI API.
"""
import openai
import json
from typing import List, Dict, Any

from .config import OPENAI_API_KEY, OPENAI_MODEL, TEMPERATURE
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
Extract only observable facts FROM USER TEXT ONLY (not from assistant section) — events or actions that could be seen, heard, or confirmed.
Do not include thoughts or feelings like 'I was afraid' or 'I felt tired'.
Keep each fact under 5 words.
"""

    emotions_context = f"""
Already existing emotions: {existing_emotions or []}
When extracting new emotions FROM USER TEXT ONLY (not from assistant section), try to match them to the existing ones if possible (by meaning, synonyms, or clear similarity). If a new emotion matches an existing one, return the existing value instead of a new variant. Only add new emotions if they are truly new and not covered by the existing ones.
"""
    
    entities_context = f"""
Already existing entities: {existing_entities or []}
When extracting new entities FROM USER TEXT ONLY (not from assistant section), try to match them to the existing ones if possible (by meaning, synonyms, or clear similarity). If a new entity matches an existing one, return the existing value instead of a new variant. Only add new entities if they are truly new and not covered by the existing ones.
"""

    # Initialize results
    facts = []
    emotions = []
    entities = []
    
    # Prepare messages for OpenAI API calls
    base_messages = [{"role": "user", "content": promptBase}]
    if chat_history and chat_history.strip():
        base_messages.append({"role": "assistant", "content": chat_history})

    # 1) Extract facts
    messages_facts = base_messages + [{"role": "system", "content": facts_context}]
    respFacts = openai.chat.completions.create(
        model=OPENAI_MODEL,
        messages=messages_facts,
        functions=[functionsSpec[0]],
        function_call="auto",
        temperature=TEMPERATURE
    )
    fc = respFacts.choices[0].message.function_call
    if fc and hasattr(fc, 'arguments'):
        facts = json.loads(fc.arguments).get("facts", [])

    # 2) Extract emotions
    messages_emotions = base_messages + [{"role": "system", "content": emotions_context}]
    respEmo = openai.chat.completions.create(
        model=OPENAI_MODEL,
        messages=messages_emotions,
        functions=[functionsSpec[1]],
        function_call="auto",
        temperature=TEMPERATURE
    )
    fc = respEmo.choices[0].message.function_call
    if fc and hasattr(fc, 'arguments'):
        emotions = json.loads(fc.arguments).get("emotions", [])

    # 3) Extract entities
    messages_entities = base_messages + [{"role": "system", "content": entities_context}]
    respEnt = openai.chat.completions.create(
        model=OPENAI_MODEL,
        messages=messages_entities,
        functions=[functionsSpec[2]],
        function_call="auto",
        temperature=TEMPERATURE
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
        )
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
