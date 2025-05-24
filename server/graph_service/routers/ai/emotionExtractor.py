import os
import json
import openai

# Assume graphiti is a configured Neo4j client passed in
# message is a dict-like object with 'uuid' and 'content' keys

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENAI_MODEL     = "gpt-4.1-mini"
TEMPERATURE      = 0.5

openai.api_key = OPENAI_API_KEY

# Define function spec for OpenAI function calling
functions_spec = [
    {
        "name": "extract_emotions",
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
    }
]

async def extractEmotionsAndStore(graphiti, message, group_id):
    """
    Extracts emotions from a message via OpenAI and stores them in Neo4j with group_id context.
    :param graphiti: Neo4j driver client
    :param message: dict with 'uuid' and 'content'
    :param group_id: identifier of the message group
    """
    prompt = (
        "Extract clear, specific emotional tones from the following message.\n"
        "Return JSON with a single key \"emotions\" whose value is an array of emotions (no commentary).\n\n"
        f"Message:\n\"\"\"{message['content']}\"\"\""
    )

    try:
        # Call OpenAI for emotion extraction
        response = await openai.ChatCompletion.acreate(
            model=OPENAI_MODEL,
            messages=[{"role": "user", "content": prompt}],
            functions=functions_spec,
            function_call="auto",
            temperature=TEMPERATURE
        )

        func_call = response.choices[0].message.function_call
        args      = json.loads(func_call.arguments)
        emotions  = args.get("emotions", [])

        # Ensure the Episodic node exists with uuid and group_id
        async with graphiti.driver.session() as session:
            await session.run(
                """
                MERGE (e:Episodic {uuid: $uuid, group_id: $group_id})
                ON CREATE SET e.group_id = $group_id
                """,
                {"uuid": message['uuid'], "group_id": group_id}
            )

            # Store each emotion relation
            for emotion in emotions:
                await session.run(
                    """
                    MERGE (em:Emotion {text: $emotion})
                    WITH em
                    MATCH (e:Episodic {uuid: $uuid, group_id: $group_id})
                    MERGE (e)-[:HAS_EMOTION]->(em)
                    """,
                    {"emotion": emotion, "uuid": message['uuid'], "group_id": group_id}
                )

        print("[Graphiti] Finished emotion extraction")

    except Exception as e:
        print(f"[Graphiti] ERROR in extract_emotions_and_store: {e}")
