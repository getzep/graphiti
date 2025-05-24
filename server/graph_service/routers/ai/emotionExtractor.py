from graph_service.config import get_settings
import openai
import os
import json

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENAI_MODEL = "gpt-4.1-mini"  # or "gpt-4", adjustable as needed
TEMPERATURE = 0.5  # model temperature

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
    prompt = f'''\
Extract clear, specific emotional tones from the following message.
Return JSON with a single key "emotions" whose value is an array of emotions (no commentary).

Message:
"""{message.content}"""
'''

    try:
        print("[Graphiti] Starting emotion extraction")
        response = openai.chat.completions.create(
            model=OPENAI_MODEL,
            messages=[{"role": "user", "content": prompt}],
            functions=functions_spec,
            function_call="auto",
            temperature=TEMPERATURE
        )
        choice = response.choices[0]
        msg = choice.message
        func_call = msg.function_call
        args = json.loads(func_call.arguments)
        emotions = args.get("emotions", [])

        for emotion in emotions:
            async with graphiti.driver.session() as session:
                await session.run("""
                    MERGE (em:Emotion {text: $emotion})
                    WITH em
                    MATCH (e:Episodic {uuid: $uuid})
                    WHERE e.group_id = $group_id
                    MERGE (e)-[:HAS_EMOTION {group_id: $group_id}]->(em)
                """, {
                    "emotion": emotion,
                    "uuid": message.uuid,
                    "group_id": group_id
                })
        print("[Graphiti] Finished emotion extraction")

    except Exception as e:
        print(f"[Graphiti] ERROR in extractEmotions: {e}")
