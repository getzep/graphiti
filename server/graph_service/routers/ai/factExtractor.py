from graph_service.config import get_settings
import openai
import os
import json

# Load settings and API key
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENAI_MODEL = "gpt-4.1-mini"
TEMPERATURE = 0.5
openai.api_key = OPENAI_API_KEY

# Define OpenAI function specifications for facts, emotions, and memories
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
        "name": "extractMemories",
        "description": "Extract vivid memories described in the message.",
        "parameters": {
            "type": "object",
            "properties": {
                "memories": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "List of user memories"
                }
            },
            "required": ["memories"]
        }
    }
]

async def extractAllAndStore(graphiti, message, groupId):
    """
    Extract facts, emotions, and memories from `message.content` using OpenAI function calls,
    then store them in Neo4j as connected nodes under Episodic(uuid).
    Prints token usage statistics for the three calls.
    """
    # Prepare prompt
    promptBase = f'''
Message content for analysis:
"""{message.content}"""
'''
    # Containers for results
    facts = []
    emotions = []
    memories = []

    try:
        # 1) Extract facts
        respFacts = openai.chat.completions.create(
            model=OPENAI_MODEL,
            messages=[{"role": "user", "content": promptBase}],
            functions=[functionsSpec[0]],
            function_call="auto",
            temperature=TEMPERATURE
        )
        facts = json.loads(respFacts.choices[0].message.function_call.arguments).get("facts", [])

        # 2) Extract emotions
        respEmo = openai.chat.completions.create(
            model=OPENAI_MODEL,
            messages=[{"role": "user", "content": promptBase}],
            functions=[functionsSpec[1]],
            function_call="auto",
            temperature=TEMPERATURE
        )
        emotions = json.loads(respEmo.choices[0].message.function_call.arguments).get("emotions", [])

        # 3) Extract memories
        respMem = openai.chat.completions.create(
            model=OPENAI_MODEL,
            messages=[{"role": "user", "content": promptBase}],
            functions=[functionsSpec[2]],
            function_call="auto",
            temperature=TEMPERATURE
        )
        memories = json.loads(respMem.choices[0].message.function_call.arguments).get("memories", [])

        # Calculate and print token usage
        inputTokens = (
            respFacts.usage.prompt_tokens +
            respEmo.usage.prompt_tokens +
            respMem.usage.prompt_tokens
        )
        outputTokens = (
            respFacts.usage.completion_tokens +
            respEmo.usage.completion_tokens +
            respMem.usage.completion_tokens
        )
        totalTokens = (
            respFacts.usage.total_tokens +
            respEmo.usage.total_tokens +
            respMem.usage.total_tokens
        )
        print(f"[Graphiti] Token usage - total: {totalTokens}, input: {inputTokens}, output: {outputTokens}")

        # 4) Store all in Neo4j
        async with graphiti.driver.session() as session:
            await session.run(
                """
                MERGE (e:Episodic {uuid: $uuid})
                ON CREATE SET e.group_id = $groupId
                WITH e

                UNWIND $emotions AS emo
                  MERGE (em:Emotion {text: emo})
                  MERGE (e)-[:HAS_EMOTION]->(em)
                WITH e

                UNWIND $facts AS fact
                  MERGE (f:Fact {text: fact})
                  MERGE (e)-[:HAS_FACT]->(f)
                WITH e

                UNWIND $memories AS mem
                  MERGE (m:Memory {text: mem})
                  MERGE (e)-[:HAS_MEMORY]->(m)
                """,
                {
                    "uuid": message.uuid,
                    "groupId": groupId,
                    "emotions": emotions,
                    "facts": facts,
                    "memories": memories
                }
            )

        print("[Graphiti] Extraction and storage complete.")

    except Exception as err:
        print(f"[Graphiti] ERROR in extractAllAndStore: {err}")

# Alias for backward compatibility
extractFactsAndStore = extractAllAndStore
