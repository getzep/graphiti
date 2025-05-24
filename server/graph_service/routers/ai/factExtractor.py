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
    # Jeśli nie ma message, nic nie rób
    if not message:
        return None

    # Prepare prompt
    promptBase = f'''
Message content for analysis:
"""{message.content}"""
'''
    # Containers for results
    facts = []
    emotions = []
    entities = []

    try:
        # 1) Extract facts
        respFacts = openai.chat.completions.create(
            model=OPENAI_MODEL,
            messages=[{"role": "user", "content": promptBase}],
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
            messages=[{"role": "user", "content": promptBase}],
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
            messages=[{"role": "user", "content": promptBase}],
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
