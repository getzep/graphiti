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
    # 1) Przygotuj prompt do OpenAI
    prompt = f"""
Extract clear, specific emotional tones from the following message.
Return JSON with a single key "emotions" whose value is an array of emotions (no commentary).

Message:
\"\"\"{message.content}\"\"\"
"""

    try:
        # 2) Wywołaj model z funkcją
        response = openai.chat.completions.create(
            model=OPENAI_MODEL,
            messages=[{"role": "user", "content": prompt}],
            functions=functions_spec,
            function_call="auto",
            temperature=TEMPERATURE
        )

        func_args = json.loads(response.choices[0].message.function_call.arguments)
        emotions = func_args.get("emotions", [])

        # 3) Jeden kontekst sesji, jedno MERGE dla Episodic, jedno UNWIND
        async with graphiti.driver.session() as session:
            await session.run(
                """
                MERGE (e:Episodic {uuid: $uuid})
                ON CREATE SET e.group_id = $group_id
                WITH e
                UNWIND $emotions AS emo
                  MERGE (em:Emotion {text: emo})
                  MERGE (e)-[:HAS_EMOTION]->(em)
                """,
                {
                    "uuid": message['uuid'],
                    "group_id": group_id,
                    "emotions": emotions
                }
            )

        print("[Graphiti] Finished emotion extraction")

    except Exception as e:
        print(f"[Graphiti] ERROR in extractEmotions: {e}")
