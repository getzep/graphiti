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
        "name": "extract_entities",
        "description": "Extract entities mentioned in the provided message.",
        "parameters": {
            "type": "object",
            "properties": {
                "entities": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "List of specific entities or objects"
                }
            },
            "required": ["entities"]
        }
    }
]

async def extractEntitiesAndStore(graphiti, message, group_id):
    prompt = f'''\
Extract only concrete entities, objects, or people that the speaker actually mentions.
Do NOT extract wishes, desires, hypothetical situations, or general statements.
If there are no such entities, return an empty array.

Message:
"""{message.content}"""
'''

    try:
        print("[Graphiti] Starting entity extraction")
         # Use function calling to get structured entities
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
        entities = args.get("entities", [])

        async with graphiti.driver.session() as session:
            await session.run(
                """
                MERGE (e:Episodic {uuid: $uuid})
                ON CREATE SET e.group_id = $group_id
                WITH e
                UNWIND $entities AS ent
                  MERGE (m:Entity {text: ent})
                  MERGE (e)-[:HAS_ENTITY]->(m)
                """,
                {"uuid": message.uuid, "group_id": group_id, "entities": entities}
            )

        print("[Graphiti] Finished entity extraction")

    except Exception as e:
        print(f"[Graphiti] ERROR in extractEntities: {e}")
