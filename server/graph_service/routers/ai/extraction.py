"""
Module for extracting facts, emotions and entities using OpenAI API, GoEmotions and FastCoref.
"""
from openai import OpenAI
import json
import logging
from typing import List, Dict, Any, Union

from .config import (
    OPENAI_API_KEY, 
    FACTS_CONFIG, 
    EMOTIONS_CONFIG, 
    ENTITIES_CONFIG
)
from .function_specs import functionsSpec
from .goemotions import extract_emotions_with_goemotions

logger = logging.getLogger(__name__)

def get_openai_client():
    """Get OpenAI client instance, initialized lazily."""
    return OpenAI(api_key=OPENAI_API_KEY)

async def extract_facts_emotions_entities(
    message_content: str, 
    existing_emotions: List[str] = None, 
    existing_entities: List[str] = None,
    chat_history: Union[List[Dict], str] = None,
    extract_emotions: bool = False
) -> Dict[str, Any]:
    """
    Extract facts, emotions, and entities from message content.
    Uses FastCoref for entity extraction and coreference resolution,
    and OpenAI for facts and emotions extraction.
    
    Args:
        message_content: Content of the message to analyze
        existing_emotions: List of existing emotions to compare against
        existing_entities: List of existing entities to compare against
        chat_history: Chat history for context - can be either:
                     - List[Dict] with 'role' and 'content' keys (new format)
                     - str (old format, for backward compatibility)
    
    Returns:
        Dictionary with:
        - facts: List of extracted facts
        - emotions: List of extracted emotions
        - entities: List of extracted entities
        - resolved_text: Text with resolved coreferences
        - usage: Token usage information
        - coreference_info: Information about coreference resolution
    """
    if not message_content or not message_content.strip():
        return {
            "facts": [], 
            "emotions": [], 
            "entities": [], 
            "resolved_text": message_content,
            "usage": {},
            "coreference_info": {}
        }

    # Step 1: Resolve coreferences and extract entities using FastCoref
    coreference_result = await extract_entities_with_coreference(
        message_content, 
        chat_history, 
        existing_entities
    )
    resolved_text = coreference_result["resolved_text"]
    entities = coreference_result["entities"]
    emotions = []
    if extract_emotions:
        emotions = await extract_emotions_with_goemotions(resolved_text)
    return {
        "facts": [],
        "emotions": emotions,
        "entities": entities,
        "resolved_text": resolved_text,
        "usage": {},
        "coreference_info": {
            "original_text": message_content,
            "clusters": coreference_result["coreference_clusters"]
        }
    }
    
    # Step 2: Extract facts and emotions using OpenAI with resolved text
    openai_results = await extract_facts_and_emotions_with_openai(
        resolved_text,
        chat_history,
        existing_emotions
    )
    
    return {
        "facts": openai_results["facts"],
        "emotions": openai_results["emotions"], 
        "entities": entities,
        "resolved_text": resolved_text,
        "usage": openai_results["usage"],
        "coreference_info": {
            "original_text": message_content,
            "clusters": coreference_result["coreference_clusters"]
        }
    }
async def extract_entities_with_coreference(
    message_content: str,
    chat_history: Union[List[Dict], str] = None,
    existing_entities: List[str] = None
) -> Dict[str, Any]:
    """
    Extract entities and resolve coreferences using FastCoref.
    
    Args:
        message_content: Current message content
        chat_history: Chat history for context
        existing_entities: Previously known entities
        
    Returns:
        Dictionary with resolved text, entities, and coreference clusters
    """
    try:        # Prepare context history for FastCoref
        context_messages = []
        
        if chat_history:
            if isinstance(chat_history, list):
                # Extract content from chat history
                for msg in chat_history:
                    if isinstance(msg, dict) and 'content' in msg:
                        content = msg.get('content', '').strip()
                        if content:
                            context_messages.append(content)
            elif isinstance(chat_history, str) and chat_history.strip():
                context_messages.append(chat_history.strip())
        
        logger.info(f"[FastCoref] context_messages for coreference: {context_messages}")
        # Get coreference resolver (import here to avoid circular imports)
        from .coreference_resolver import get_coreference_resolver
        resolver = get_coreference_resolver()
        
        if resolver.is_available():
            # Use FastCoref for coreference resolution and entity extraction
            result = resolver.resolve_coreferences_and_extract_entities(
                text=message_content,
                context_history=context_messages,
                existing_entities=existing_entities
            )
        else:
            # Fallback if FastCoref is not available
            logger.warning("FastCoref not available, using fallback entity extraction")
            result = {
                "resolved_text": message_content,
                "entities": existing_entities or [],
                "coreference_clusters": []
            }
            
        return result
        
    except Exception as e:
        logger.error(f"Error in coreference resolution: {e}")
        return {
            "resolved_text": message_content,
            "entities": existing_entities or [],
            "coreference_clusters": []
        }


async def extract_facts_and_emotions_with_openai(
    message_content: str,
    chat_history: Union[List[Dict], str] = None,
    existing_emotions: List[str] = None
) -> Dict[str, Any]:
    """
    Extract facts using OpenAI API and emotions using GoEmotions model.
    
    Args:
        message_content: Content to analyze (should be resolved text)
        chat_history: Chat history for context
        existing_emotions: Previously known emotions
        
    Returns:
        Dictionary with facts, emotions, and usage information
    """
    facts = []
    emotions = []
    
    # Prepare base messages for OpenAI (only for facts)
    base_messages = prepare_openai_messages(message_content, chat_history)
    
    # Prepare context prompts
    facts_context = get_facts_extraction_prompt()
    
    try:
        # Extract facts using OpenAI
        facts_result = await call_openai_for_facts(base_messages, facts_context)
        facts = facts_result.get("facts", [])
        
        # Extract emotions using GoEmotions
        emotions = await extract_emotions_with_goemotions(message_content, existing_emotions)
        
        # Only facts usage since emotions are processed locally
        usage = facts_result.get("usage", {})
        
        return {
            "facts": facts,
            "emotions": emotions,
            "usage": usage
        }
        
    except Exception as e:
        logger.error(f"Error in extraction: {e}")
        return {
            "facts": [],
            "emotions": [],
            "usage": {}
        }


def prepare_openai_messages(
    message_content: str, 
    chat_history: Union[List[Dict], str] = None
) -> List[Dict]:
    """
    Prepare messages for OpenAI API calls.
    
    Args:
        message_content: Current message content
        chat_history: Chat history for context
        
    Returns:
        List of formatted messages for OpenAI
    """
    base_messages = []
    
    # Add chat history to base messages if it exists
    if chat_history:
        if isinstance(chat_history, list) and len(chat_history) > 0:
            # Convert chat_history list to OpenAI messages format
            for msg in chat_history:
                if isinstance(msg, dict) and 'role' in msg and 'content' in msg:
                    # Only add non-empty messages
                    if msg['content'] and msg['content'].strip():
                        base_messages.append({
                            "role": msg['role'], 
                            "content": msg['content']
                        })
        elif isinstance(chat_history, str) and chat_history.strip():
            # Backward compatibility for string chat_history
            base_messages.append({"role": "assistant", "content": chat_history})
    
    # Add the current user message
    base_messages.append({"role": "user", "content": message_content})
    
    return base_messages


def get_facts_extraction_prompt() -> str:
    """Get the prompt for facts extraction."""
    return """
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


def get_emotions_extraction_prompt(existing_emotions: List[str] = None) -> str:
    """Get the prompt for emotions extraction."""
    return f"""
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


async def call_openai_for_facts(base_messages: List[Dict], facts_context: str) -> Dict[str, Any]:
    """Call OpenAI API for facts extraction."""
    messages_facts = base_messages + [{"role": "system", "content": facts_context}]
    
    openai_client = get_openai_client()
    resp_facts = openai_client.chat.completions.create(
        model=FACTS_CONFIG.model,
        messages=messages_facts,
        functions=[functionsSpec[0]],
        function_call="auto",
        temperature=FACTS_CONFIG.temperature
    )
    
    facts = []
    fc = resp_facts.choices[0].message.function_call
    if fc and hasattr(fc, 'arguments'):
        facts = json.loads(fc.arguments).get("facts", [])
    
    return {
        "facts": facts,
        "usage": resp_facts.usage
    }


async def call_openai_for_emotions(base_messages: List[Dict], emotions_context: str) -> Dict[str, Any]:
    """Call OpenAI API for emotions extraction."""
    messages_emotions = base_messages + [{"role": "system", "content": emotions_context}]
    
    openai_client = get_openai_client()
    resp_emo = openai_client.chat.completions.create(
        model=EMOTIONS_CONFIG.model,
        messages=messages_emotions,
        functions=[functionsSpec[1]],
        function_call="auto",
        temperature=EMOTIONS_CONFIG.temperature
    )
    
    emotions = []
    fc = resp_emo.choices[0].message.function_call
    if fc and hasattr(fc, 'arguments'):
        emotions = json.loads(fc.arguments).get("emotions", [])
    
    return {
        "emotions": emotions,
        "usage": resp_emo.usage
    }


def calculate_combined_usage(facts_result: Dict, emotions_result: Dict) -> Dict[str, Any]:
    """Calculate combined token usage from facts and emotions extraction."""
    facts_usage = facts_result.get("usage")
    emotions_usage = emotions_result.get("usage")
    
    if not facts_usage or not emotions_usage:
        return {}
    
    return {
        "input_tokens": (
            facts_usage.prompt_tokens +
            emotions_usage.prompt_tokens
        ),
        "output_tokens": (
            facts_usage.completion_tokens +
            emotions_usage.completion_tokens
        ),
        "total_tokens": (
            facts_usage.total_tokens +
            emotions_usage.total_tokens
        ),
        "models": {
            "facts": FACTS_CONFIG.model,
            "emotions": EMOTIONS_CONFIG.model,
        },
        "temperatures": {
            "facts": FACTS_CONFIG.temperature,
            "emotions": EMOTIONS_CONFIG.temperature,
        }
    }


# Keep the old function for backward compatibility
async def extract_facts_emotions_entities_legacy(
    message_content: str, 
    existing_emotions: List[str] = None, 
    existing_entities: List[str] = None,
    chat_history = None
) -> Dict[str, List[str]]:
    """
    Legacy function for backward compatibility.
    This maintains the old interface while using the new implementation.
    """
    result = await extract_facts_emotions_entities(
        message_content=message_content,
        existing_emotions=existing_emotions,
        existing_entities=existing_entities,
        chat_history=chat_history
    )
    
    # Return in old format
    return {
        "facts": result["facts"],
        "emotions": result["emotions"], 
        "entities": result["entities"],
        "usage": result["usage"]
    }
