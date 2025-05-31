# Plan refaktoryzacji modułu factExtractor.py

Aktualny plik `factExtractor.py` jest zbyt duży i monolityczny. Proponuję podzielić go na następujące komponenty:

## Struktura proponowanych plików

1. `config.py` - konfiguracja OpenAI API
2. `function_specs.py` - definicje funkcji dla API OpenAI
3. `extraction.py` - logika ekstrakcji danych
4. `neo4j_operations.py` - operacje na bazie danych Neo4j
5. `factExtractor.py` - główny moduł integrujący pozostałe komponenty

## Zawartość plików

### 1. config.py
```python
"""
Configuration for AI services used in fact extraction.
"""
import os

# Load settings and API key
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENAI_MODEL = "gpt-4.1-mini"
TEMPERATURE = 0.25
```

### 2. function_specs.py
```python
"""
Function specifications for OpenAI API calls.
"""

# Define OpenAI function specifications for facts, emotions, and entities
functionsSpec = [
   {
        "name": "extractFacts",
        "description": "List of objective facts based on observable events or actions. Each fact must be concise (max 5 words) and exclude feelings or thoughts.",
        "parameters": {
            "type": "object",
            "properties": {
                "facts": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "List of short factual statements, each no longer than 5 words"
                }
            },
            "required": ["facts"]
        }
    },
    {
        "name": "extractEmotions",
        "description": "Extract emotional tones from the provided message.",
        "parameters": {
            "type": "object",
            "properties": {
                "emotions": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "List of emotional tones"
                }
            },
            "required": ["emotions"]
        }
    },
   {
        "name": "extractEntities",
        "description": "Identify specific people mentioned in the message.",
        "parameters": {
            "type": "object",
            "properties": {
                "entities": {
                    "type": "array",
                    "items": { "type": "string" },
                    "description": "List of specific people mentioned in the message. Include names or clear roles (e.g. 'my neighbor', 'dad', 'Marta'). Do not include vague pronouns like 'she', 'someone', 'they'."
                }
            },
            "required": ["entities"]
        }
    }
]
```

### 3. extraction.py
```python
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
```

### 4. neo4j_operations.py
```python
"""
Neo4j database operations for fact extraction.
"""
from typing import Dict, List, Any, Optional

async def get_existing_data(session, group_id: str) -> Dict[str, List[str]]:
    """
    Retrieve existing facts, emotions, and entities for a given group_id.
    """
    # Initialize results
    existing_facts = []
    existing_emotions = []
    existing_entities = []
    
    # Fetch existing facts
    result_facts = await session.run(
        """
        MATCH (fact_node:Fact)<-[r:IS_FACT]-(:Episodic)
        WHERE r.group_id = $group_id
        RETURN collect(DISTINCT {text: fact_node.text, count: COALESCE(fact_node.count, 0)}) AS fact_nodes_for_group
        """,
        {"group_id": group_id}
    )
    record_facts = await result_facts.single()
    if record_facts and record_facts["fact_nodes_for_group"]:
        fact_nodes = record_facts["fact_nodes_for_group"]
        existing_facts = [fn["text"] for fn in fact_nodes if fn and fn.get("text")]
    
    # Fetch existing emotions
    result_emotions = await session.run(
        """
        MATCH (emotion_node:Emotion)<-[r:HAS_EMOTION]-(:Episodic)
        WHERE r.group_id = $group_id
        RETURN collect(DISTINCT {text: emotion_node.text, count: COALESCE(emotion_node.count, 0)}) AS emotion_nodes_for_group
        """,
        {"group_id": group_id}
    )
    record_emotions = await result_emotions.single()
    if record_emotions and record_emotions["emotion_nodes_for_group"]:
        emotion_nodes = record_emotions["emotion_nodes_for_group"]
        existing_emotions = [en["text"] for en in emotion_nodes if en and en.get("text")]

    # Fetch existing entities
    result_entities = await session.run(
        """
        MATCH (entity_node:Entity)<-[r:HAS_ENTITY]-(:Episodic)
        WHERE r.group_id = $group_id
        RETURN collect(DISTINCT {text: entity_node.text, count: COALESCE(entity_node.count, 0)}) AS entity_nodes_for_group
        """,
        {"group_id": group_id}
    )
    record_entities = await result_entities.single()
    if record_entities and record_entities["entity_nodes_for_group"]:
        entity_nodes = record_entities["entity_nodes_for_group"]
        existing_entities = [mn["text"] for mn in entity_nodes if mn and mn.get("text")]
        
    return {
        "facts": existing_facts,
        "emotions": existing_emotions,
        "entities": existing_entities
    }

async def store_extracted_data(
    session, 
    uuid: str, 
    group_id: str, 
    facts: List[str], 
    emotions: List[str], 
    entities: List[str], 
    shirt_slug: str
) -> None:
    """
    Store extracted facts, emotions, and entities in Neo4j.
    """
    await session.run(
        """
        MERGE (e:Episodic {uuid: $uuid})
        ON CREATE SET e.group_id = $group_id
        WITH e
        MERGE (s:Shirt {slug: $shirt_slug})
        WITH e, s
        MERGE (e)-[:CONNECTED_TO]->(s)
        WITH e

        UNWIND $emotions AS emo
          MERGE (em:Emotion {text: emo})
          ON CREATE SET em.count = 1
          ON MATCH SET em.count = COALESCE(em.count, 0) + 1
          MERGE (e)-[rel:HAS_EMOTION {group_id: $group_id, shirt_slug: $shirt_slug}]->(em)
        WITH e

        UNWIND $facts AS fact
          MERGE (f:Fact {text: fact})
          ON CREATE SET f.count = 1
          ON MATCH SET f.count = COALESCE(f.count, 0) + 1
          MERGE (e)-[rel:IS_FACT {group_id: $group_id, shirt_slug: $shirt_slug}]->(f)
        WITH e

        UNWIND $entities AS ent
          MERGE (m:Entity {text: ent})
          ON CREATE SET m.count = 1
          ON MATCH SET m.count = COALESCE(m.count, 0) + 1
          MERGE (e)-[rel:HAS_ENTITY {group_id: $group_id, shirt_slug: $shirt_slug}]->(m)
        """,
        {
            "uuid": uuid,
            "group_id": group_id,
            "emotions": emotions,
            "facts": facts,
            "entities": entities,
            "shirt_slug": shirt_slug
        }
    )

async def get_relationships_data(session, group_id: str) -> Dict[str, List[Dict[str, Any]]]:
    """
    Get relationships between facts, emotions, and entities.
    """
    facts_connected_to_entities = []
    facts_connected_to_emotions = []
    emotions_connected_to_entities = []
    
    # Facts connected to entities
    result_facts_entities = await session.run(
        """
        MATCH (fact_node:Fact)<-[rf:IS_FACT]-(ep:Episodic)-[re:HAS_ENTITY]->(entity_node:Entity)
        WHERE rf.group_id = $group_id AND re.group_id = $group_id
        RETURN collect(DISTINCT {
            fact: fact_node.text, 
            entity: entity_node.text, 
            fact_count: COALESCE(fact_node.count, 0),
            entity_count: COALESCE(entity_node.count, 0)
        }) AS facts_with_entities
        """,
        {"group_id": group_id}
    )
    record_facts_entities = await result_facts_entities.single()
    if record_facts_entities and record_facts_entities["facts_with_entities"]:
        facts_connected_to_entities = record_facts_entities["facts_with_entities"]

    # Facts connected to emotions
    result_facts_emotions = await session.run(
        """
        MATCH (fact_node:Fact)<-[rf:IS_FACT]-(ep:Episodic)-[rem:HAS_EMOTION]->(emotion_node:Emotion)
        WHERE rf.group_id = $group_id AND rem.group_id = $group_id
        RETURN collect(DISTINCT {
            fact: fact_node.text, 
            emotion: emotion_node.text,
            fact_count: COALESCE(fact_node.count, 0),
            emotion_count: COALESCE(emotion_node.count, 0)
        }) AS facts_with_emotions
        """,
        {"group_id": group_id}
    )
    record_facts_emotions = await result_facts_emotions.single()
    if record_facts_emotions and record_facts_emotions["facts_with_emotions"]:
        facts_connected_to_emotions = record_facts_emotions["facts_with_emotions"]

    # Emotions connected to entities
    result_emotions_entities = await session.run(
        """
        MATCH (emotion_node:Emotion)<-[rem:HAS_EMOTION]-(ep:Episodic)-[re:HAS_ENTITY]->(entity_node:Entity)
        WHERE rem.group_id = $group_id AND re.group_id = $group_id
        RETURN collect(DISTINCT {
            emotion: emotion_node.text, 
            entity: entity_node.text,
            emotion_count: COALESCE(emotion_node.count, 0),
            entity_count: COALESCE(entity_node.count, 0)
        }) AS emotions_with_entities
        """,
        {"group_id": group_id}
    )
    record_emotions_entities = await result_emotions_entities.single()
    if record_emotions_entities and record_emotions_entities["emotions_with_entities"]:
        emotions_connected_to_entities = record_emotions_entities["emotions_with_entities"]
    
    return {
        "facts_entities": facts_connected_to_entities,
        "facts_emotions": facts_connected_to_emotions,
        "emotions_entities": emotions_connected_to_entities
    }

async def get_top_items(session, group_id: str) -> Dict[str, List[Dict[str, Any]]]:
    """
    Get top facts, emotions, and entities by count.
    """
    # Top facts
    result_top_facts = await session.run(
        """
        MATCH (fact_node:Fact)<-[r:IS_FACT]-(:Episodic)
        WHERE r.group_id = $group_id
        WITH fact_node.text AS text, MAX(fact_node.count) AS count
        ORDER BY count DESC LIMIT 5
        RETURN collect({text: text, count: count}) AS top_facts
        """,
        {"group_id": group_id}
    )
    record_top_facts = await result_top_facts.single()
    top_facts = record_top_facts["top_facts"] if record_top_facts else []
    
    # Top emotions
    result_top_emotions = await session.run(
        """
        MATCH (emotion_node:Emotion)<-[r:HAS_EMOTION]-(:Episodic)
        WHERE r.group_id = $group_id
        WITH emotion_node.text AS text, MAX(emotion_node.count) AS count
        ORDER BY count DESC LIMIT 5
        RETURN collect({text: text, count: count}) AS top_emotions
        """,
        {"group_id": group_id}
    )
    record_top_emotions = await result_top_emotions.single()
    top_emotions = record_top_emotions["top_emotions"] if record_top_emotions else []
    
    # Top entities
    result_top_entities = await session.run(
        """
        MATCH (entity_node:Entity)<-[r:HAS_ENTITY]-(:Episodic)
        WHERE r.group_id = $group_id
        WITH entity_node.text AS text, MAX(entity_node.count) AS count
        ORDER BY count DESC LIMIT 5
        RETURN collect({text: text, count: count}) AS top_entities
        """,
        {"group_id": group_id}
    )
    record_top_entities = await result_top_entities.single()
    top_entities = record_top_entities["top_entities"] if record_top_entities else []
    
    return {
        "top_facts": top_facts,
        "top_emotions": top_emotions,
        "top_entities": top_entities
    }
```

### 5. factExtractor.py (Główny plik)
```python
"""
Main fact extractor module that coordinates the extraction and storage of facts, emotions, and entities.
"""
import json
import logging
from typing import Dict, Any, Optional

# Import components
from .config import OPENAI_API_KEY
from .extraction import extract_facts_emotions_entities
from .neo4j_operations import (
    get_existing_data,
    store_extracted_data,
    get_relationships_data,
    get_top_items
)

# Configure logger
logger = logging.getLogger(__name__)

async def extractAllAndStore(graphiti, message, group_id, chat_history, shirt_slug):
    """
    Extract facts, emotions, and entities from `message.content` using OpenAI function calls,
    then store them in Neo4j as connected nodes under Episodic(uuid).
    """
    # Jeśli nie ma message lub message.content, nic nie rób
    if not message or not hasattr(message, 'content') or not message.content or not message.content.strip():
        return None

    try:
        # 1. Pobierz istniejące fakty, emocje i encje dla danego group_id
        async with graphiti.driver.session() as session:
            existing_data = await get_existing_data(session, group_id)
            
        # 2. Wykonaj ekstrakcję faktów, emocji i encji
        extraction_results = await extract_facts_emotions_entities(
            message_content=message.content,
            existing_emotions=existing_data["emotions"],
            existing_entities=existing_data["entities"],
            chat_history=chat_history
        )
        
        facts = extraction_results["facts"]
        emotions = extraction_results["emotions"]
        entities = extraction_results["entities"]
        
        # Log token usage
        usage = extraction_results["usage"]
        logger.info(
            f"[Graphiti] Token usage - Input: {usage['input_tokens']}, "
            f"Output: {usage['output_tokens']}, Total: {usage['total_tokens']}"
        )
        
        # 3. Store data in Neo4j
        async with graphiti.driver.session() as session:
            # Store main data
            await store_extracted_data(
                session, 
                message.uuid, 
                group_id, 
                facts, 
                emotions, 
                entities, 
                shirt_slug
            )
            
            # Fetch relationship data (not used but could be useful for future analysis)
            relationships = await get_relationships_data(session, group_id)
            
        # 4. Get top facts, emotions, and entities
        async with graphiti.driver.session() as session:
            top_items = await get_top_items(session, group_id)
            
        return top_items
        
    except Exception as e:
        logger.error(f"[Graphiti] Error in extractAllAndStore: {str(e)}")
        return {
            "top_facts": [],
            "top_emotions": [],
            "top_entities": [],
            "error": str(e)
        }
```

## Korzyści z refaktoryzacji

1. **Lepsza organizacja kodu**: Każdy moduł ma swoją specyficzną odpowiedzialność.
2. **Łatwiejsze testowanie**: Mniejsze moduły są łatwiejsze do testowania.
3. **Lepsza czytelność**: Łatwiej zrozumieć, co robi każda część kodu.
4. **Możliwość ponownego użycia komponentów**: Poszczególne moduły mogą być używane w innych częściach aplikacji.

## Plan wdrożenia

1. Stworzyć nowe pliki zgodnie z powyższą strukturą
2. Zmodyfikować istniejący plik `factExtractor.py` na końcu (aby zachować kompatybilność)
3. Zaktualizować importy w pliku głównym
4. Dodać testy dla nowej struktury
5. Stopniowo modyfikować miejsca, które używają `factExtractor.py`
