from graph_service.config import get_settings
import openai
import os
import json

# Load settings and API key
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENAI_MODEL = "gpt-4.1-mini"
TEMPERATURE = 0.25
openai.api_key = OPENAI_API_KEY

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

async def extractAllAndStore(graphiti, message, group_id, chat_history, shirt_slug):
    """
    Extract facts, emotions, and entities from `message.content` using OpenAI function calls,
    then store them in Neo4j as connected nodes under Episodic(uuid).
    Prints token usage statistics for the three calls and returns the total token usage.
    """
    # Jeśli nie ma message lub message.content, nic nie rób
    if not message or not hasattr(message, 'content') or not message.content or not message.content.strip():
        return None

    # 1. Pobierz istniejące fakty, emocje i encje dla danego group_id
    existing_facts = []
    existing_emotions = []
    existing_entities = []
    facts_connected_to_entities = [] # Inicjalizacje pozostają na górze funkcji
    facts_connected_to_emotions = []
    emotions_connected_to_entities = []

    async with graphiti.driver.session() as session:
        # Pobierz istniejące fakty dla grupy
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
        
        # Pobierz istniejące emocje dla grupy
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

        # Pobierz istniejące encje dla grupy
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

        # Usunięto stąd blok pobierający facts_connected_to_entities, facts_connected_to_emotions, emotions_connected_to_entities.
        # Został on przeniesiony na koniec funkcji

    # 2. Przygotuj bazowy prompt tylko z treścią wiadomości
    promptBase = f'''
Message content for analysis:
"""{message.content}"""
'''

    # 3. faktów nie porównujemy z istniejącymi (za dużo ich może być),
    facts_context = f"""
Extract only observable facts FROM USER TEXT ONLY (not from assistant section) — events or actions that could be seen, heard, or confirmed.
Do not include thoughts or feelings like 'I was afraid' or 'I felt tired'.
Keep each fact under 5 words.
"""

    emotions_context = f"""
Already existing emotions: {existing_emotions}
When extracting new emotions FROM USER TEXT ONLY (not from assistant section), try to match them to the existing ones if possible (by meaning, synonyms, or clear similarity). If a new emotion matches an existing one, return the existing value instead of a new variant. Only add new emotions if they are truly new and not covered by the existing ones.
"""
    entities_context = f"""
Already existing entities: {existing_entities}
When extracting new entities FROM USER TEXT ONLY (not from assistant section), try to match them to the existing ones if possible (by meaning, synonyms, or clear similarity). If a new entity matches an existing one, return the existing value instead of a new variant. Only add new entities if they are truly new and not covered by the existing ones.
"""

    # Containers for results
    facts = []
    emotions = []
    entities = []

    try:
        # Prepare messages for OpenAI API calls
        base_messages = [{"role": "user", "content": promptBase}]
        print(f"[Graphiti] Extracting facts, emotions, and entities. History: {chat_history}, Shirt Slug: {shirt_slug}")
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
        facts = []
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
        emotions = []
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
        entities = []
        fc = respEnt.choices[0].message.function_call
        if fc and hasattr(fc, 'arguments'):
            entities = json.loads(fc.arguments).get("entities", [])

        # Calculate and print token usage
        inputTokens = (
            respFacts.usage.prompt_tokens +
            respEmo.usage.prompt_tokens +
            respEnt.usage.prompt_tokens
        )
        outputTokens = (
            respFacts.usage.completion_tokens +
            respEmo.usage.completion_tokens +
            respEnt.usage.completion_tokens
        )
        totalTokens = (
            respFacts.usage.total_tokens +
            respEmo.usage.total_tokens +
            respEnt.usage.total_tokens
        )

        # 4) Store all in Neo4j
        # Ensure all lists are unique
        emotions = list(set(emotions))
        facts = list(set(facts))
        entities = list(set(entities))
        async with graphiti.driver.session() as session:
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
                    "uuid": message.uuid,
                    "group_id": group_id,
                    "emotions": emotions,
                    "facts": facts,
                    "entities": entities,
                    "shirt_slug": shirt_slug
                }
            )

            # Pobierz fakty połączone z osobami (entities) - PRZENIESIONE TUTAJ
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
            else:
                facts_connected_to_entities = []

            # Pobierz fakty powiązane z emocjami - PRZENIESIONE TUTAJ
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
            else:
                facts_connected_to_emotions = []

            # Pobierz emocje powiązane z osobami (entities) - PRZENIESIONE TUTAJ
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
            else:
                emotions_connected_to_entities = []

        # Przygotuj informacje o licznikach dla emotions, entities i facts
        async with graphiti.driver.session() as session:
            # Pobierz liczniki dla emotions
            result_emotions_counts = await session.run(
                """
                MATCH (emotion_node:Emotion)<-[r:HAS_EMOTION]-(:Episodic)
                WHERE r.group_id = $group_id AND emotion_node.text IN $emotions
                RETURN DISTINCT emotion_node.text AS text, emotion_node.count AS count
                """,
                {"group_id": group_id, "emotions": emotions}
            )
            emotions_with_counts = []
            unique_emotions = set()  # Zbiór unikalnych tekstów emocji
            async for record in result_emotions_counts:
                emotion_text = record["text"]
                if emotion_text not in unique_emotions:
                    emotions_with_counts.append({
                        "text": emotion_text, 
                        "count": record["count"]
                    })
                    unique_emotions.add(emotion_text)
            
            # Pobierz liczniki dla entities
            result_entities_counts = await session.run(
                """
                MATCH (entity_node:Entity)<-[r:HAS_ENTITY]-(:Episodic)
                WHERE r.group_id = $group_id AND entity_node.text IN $entities
                RETURN DISTINCT entity_node.text AS text, entity_node.count AS count
                """,
                {"group_id": group_id, "entities": entities}
            )
            entities_with_counts = []
            unique_entities = set()  # Zbiór unikalnych tekstów encji
            async for record in result_entities_counts:
                entity_text = record["text"]
                if entity_text not in unique_entities:
                    entities_with_counts.append({
                        "text": entity_text, 
                        "count": record["count"]
                    })
                    unique_entities.add(entity_text)
            
            # Pobierz liczniki dla facts
            result_facts_counts = await session.run(
                """
                MATCH (fact_node:Fact)<-[r:IS_FACT]-(:Episodic)
                WHERE r.group_id = $group_id AND fact_node.text IN $facts
                RETURN DISTINCT fact_node.text AS text, fact_node.count AS count
                """,
                {"group_id": group_id, "facts": facts}
            )
            facts_with_counts = []
            unique_facts = set()  # Zbiór unikalnych tekstów faktów
            async for record in result_facts_counts:
                fact_text = record["text"]
                if fact_text not in unique_facts:
                    facts_with_counts.append({
                        "text": fact_text, 
                        "count": record["count"]
                    })
                    unique_facts.add(fact_text)
       