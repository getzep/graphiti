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
        "name": "extract_relations",
        "description": "Extract relationships between the message subject and other entities (people, animals, objects)",
        "parameters": {
            "type": "object",
            "properties": {
                "relations": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "List of relationship statements"
                }
            },
            "required": ["relations"]
        }
    }
]

async def extractRelationsAndStore(graphiti, message, group_id):
    prompt = f'''\
Extract clear, specific relationship statements between the message subject and other entities (people, animals, objects).
Return JSON with a single key "relations" whose value is an array of short relationship descriptions (no commentary).

Message:
"""{message.content}"""
'''  

    try:
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
        relations = args.get("relations", [])

        async with graphiti.driver.session() as session:
            await session.run(
                """
                MERGE (e:Episodic {uuid: $uuid})
                ON CREATE SET e.group_id = $group_id
                WITH e
                UNWIND $relations AS rel
                  MERGE (r:Relation {text: rel})
                  MERGE (e)-[:HAS_RELATION]->(r)
                """,
                {"uuid": message.uuid, "group_id": group_id, "relations": relations}
            )
        print("[Graphiti] Finished relation extraction")

    except Exception as e:
        print(f"[Graphiti] ERROR in extractRelations: {e}")
