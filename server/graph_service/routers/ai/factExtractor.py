from graph_service.config import get_settings
import openai
import os
import json

# Load settings and API key
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENAI_MODEL = "gpt-4.1-mini"
TEMPERATURE = 0.0
openai.api_key = OPENAI_API_KEY

# Define OpenAI function specifications for facts, emotions, and entities
functionsSpec = [
    {
        "name": "extractFacts",
        "description": "Extract factual statements from the provided message.",
        "parameters": {
            "type": "object",
            "properties": {
                "facts": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "List of short factual statements"
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

async def extractAllAndStore(graphiti, message, groupId):
    """
    Extract facts, emotions, and entities from `message.content` using OpenAI function calls,
    then store them in Neo4j as connected nodes under Episodic(uuid).
    Prints token usage statistics for the three calls and returns the total token usage.
    """
    # Jeśli nie ma message lub message.content, nic nie rób
    if not message or not hasattr(message, 'content') or not message.content or not message.content.strip():
        return None

    # 1. Pobierz istniejące fakty, emocje i encje dla danego groupId
    existing_facts = []
    existing_emotions = []
    existing_entities = []
    print(f"[DEBUG] Attempting to fetch existing data for groupId: {groupId}") # Zmieniono log
    async with graphiti.driver.session() as session:
        # Pobierz istniejące fakty dla grupy
        result_facts = await session.run(
            """
            MATCH (fact_node:Fact)<-[r:IS_FACT]-(:Episodic)
            WHERE r.group_id = $groupId
            RETURN collect(DISTINCT fact_node) AS fact_nodes_for_group
            """,
            {"groupId": groupId}
        )
        record_facts = await result_facts.single()
        if record_facts and record_facts["fact_nodes_for_group"]:
            fact_nodes = record_facts["fact_nodes_for_group"]
            print(f"[DEBUG] Fact nodes from DB for group: {fact_nodes}") # Zmieniono log
            existing_facts = [fn["text"] for fn in fact_nodes if fn and fn.get("text")]
        
        # Pobierz istniejące emocje dla grupy
        result_emotions = await session.run(
            """
            MATCH (emotion_node:Emotion)<-[r:HAS_EMOTION]-(:Episodic)
            WHERE r.group_id = $groupId
            RETURN collect(DISTINCT emotion_node) AS emotion_nodes_for_group
            """,
            {"groupId": groupId}
        )
        record_emotions = await result_emotions.single()
        if record_emotions and record_emotions["emotion_nodes_for_group"]:
            emotion_nodes = record_emotions["emotion_nodes_for_group"]
            print(f"[DEBUG] Emotion nodes from DB for group: {emotion_nodes}") # Zmieniono log
            existing_emotions = [en["text"] for en in emotion_nodes if en and en.get("text")]

        # Pobierz istniejące encje dla grupy
        result_entities = await session.run(
            """
            MATCH (entity_node:Entity)<-[r:HAS_ENTITY]-(:Episodic)
            WHERE r.group_id = $groupId
            RETURN collect(DISTINCT entity_node) AS entity_nodes_for_group
            """,
            {"groupId": groupId}
        )
        record_entities = await result_entities.single()
        if record_entities and record_entities["entity_nodes_for_group"]:
            entity_nodes = record_entities["entity_nodes_for_group"]
            print(f"[DEBUG] Entity nodes from DB for group: {entity_nodes}") # Zmieniono log
            existing_entities = [mn["text"] for mn in entity_nodes if mn and mn.get("text")]

    # Usunięto logi dotyczące episodic_node, bo nie jest już bezpośrednio pobierany w tym bloku
    # print(f"[DEBUG] Episodic node found: {episodic_node}") 
            
    print(f"[DEBUG] Populated existing_facts for group: {existing_facts}") # Zmieniono log
    print(f"[DEBUG] Populated existing_emotions for group: {existing_emotions}") # Zmieniono log
    print(f"[DEBUG] Populated existing_entities for group: {existing_entities}") # Zmieniono log
    # Usunięto warunek else, który logował brak rekordu dla uuid, bo teraz szukamy po groupId
    # else:
    #     print("[DEBUG] No record found in DB for this uuid (Episodic node not found).")

    # 2. Przygotuj bazowy prompt tylko z treścią wiadomości
    promptBase = f'''
Message content for analysis:
"""{message.content}"""
'''

    # 3. Przygotuj osobne instrukcje i kontekst do każdej funkcji
    facts_context = f"""
Already existing facts: {existing_facts}
When extracting new facts, try to match them to the existing ones if possible (by meaning, synonyms, or clear similarity). If a new fact matches an existing one, return the existing value instead of a new variant. Only add new facts if they are truly new and not covered by the existing ones.
"""
    emotions_context = f"""
Already existing emotions: {existing_emotions}
When extracting new emotions, try to match them to the existing ones if possible (by meaning, synonyms, or clear similarity). If a new emotion matches an existing one, return the existing value instead of a new variant. Only add new emotions if they are truly new and not covered by the existing ones.
"""
    entities_context = f"""
Already existing entities: {existing_entities}
When extracting new entities, try to match them to the existing ones if possible (by meaning, synonyms, or clear similarity). If a new entity matches an existing one, return the existing value instead of a new variant. Only add new entities if they are truly new and not covered by the existing ones.
"""

    # Containers for results
    facts = []
    emotions = []
    entities = []

    try:
        # 1) Extract facts
        respFacts = openai.chat.completions.create(
            model=OPENAI_MODEL,
            messages=[
                {"role": "user", "content": promptBase},
                {"role": "system", "content": facts_context}
            ],
            functions=[functionsSpec[0]],
            function_call="auto",
            temperature=TEMPERATURE
        )
        facts = []
        fc = respFacts.choices[0].message.function_call
        if fc and hasattr(fc, 'arguments'):
            facts = json.loads(fc.arguments).get("facts", [])

        # 2) Extract emotions
        respEmo = openai.chat.completions.create(
            model=OPENAI_MODEL,
            messages=[
                {"role": "user", "content": promptBase},
                {"role": "system", "content": emotions_context}
            ],
            functions=[functionsSpec[1]],
            function_call="auto",
            temperature=TEMPERATURE
        )
        emotions = []
        fc = respEmo.choices[0].message.function_call
        if fc and hasattr(fc, 'arguments'):
            emotions = json.loads(fc.arguments).get("emotions", [])

        # 3) Extract entities
        respEnt = openai.chat.completions.create(
            model=OPENAI_MODEL,
            messages=[
                {"role": "user", "content": promptBase},
                {"role": "system", "content": entities_context}
            ],
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
        # print(f"[Graphiti] Token usage - total: {totalTokens}, input: {inputTokens}, output: {outputTokens}")

        # 4) Store all in Neo4j
        # Ensure all lists are unique
        emotions = list(set(emotions))
        facts = list(set(facts))
        entities = list(set(entities))
        async with graphiti.driver.session() as session:
            await session.run(
                """
                MERGE (e:Episodic {uuid: $uuid})
                ON CREATE SET e.group_id = $groupId
                WITH e

                UNWIND $emotions AS emo
                  MERGE (em:Emotion {text: emo})
                  MERGE (e)-[rel:HAS_EMOTION {group_id: $groupId}]->(em)
                WITH e

                UNWIND $facts AS fact
                  MERGE (f:Fact {text: fact})
                  MERGE (e)-[rel:IS_FACT {group_id: $groupId}]->(f)
                WITH e

                UNWIND $entities AS ent
                  MERGE (m:Entity {text: ent})
                  MERGE (e)-[rel:HAS_ENTITY {group_id: $groupId}]->(m)
                """,
                {
                    "uuid": message.uuid,
                    "groupId": groupId,
                    "emotions": emotions,
                    "facts": facts,
                    "entities": entities
                }
            )

        # print("[Graphiti] Extraction and storage complete.")

        return {
            "total_tokens": totalTokens,
            "input_tokens": inputTokens,
            "output_tokens": outputTokens,
            "model": OPENAI_MODEL,
            "temperature": TEMPERATURE
        }

    except Exception as err:
        print(f"[Graphiti] ERROR in extractAllAndStore: {err}")
        return None

# Alias for backward compatibility
extractFactsAndStore = extractAllAndStore
