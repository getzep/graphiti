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
        "name": "extract_memories",
        "description": "Extract recollections or memories mentioned in the provided message.",
        "parameters": {
            "type": "object",
            "properties": {
                "memories": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "List of specific memories or recollections"
                }
            },
            "required": ["memories"]
        }
    }
]

async def extractMemoriesAndStore(graphiti, message, group_id):
    prompt = f'''\
Extract only past personal experiences or specific events that the speaker actually lived through.
Do NOT extract wishes, desires, hypothetical situations, or general statements.
If there are no such memories, return an empty array.

Message:
"""{message.content}"""
'''

    try:
        print("[Graphiti] Starting memory extraction")
         # Use function calling to get structured memories
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
        memories = args.get("memories", [])

        async with graphiti.driver.session() as session:
            await session.run(
                """
                MERGE (e:Episodic {uuid: $uuid})
                ON CREATE SET e.group_id = $group_id
                WITH e
                UNWIND $memories AS mem
                  MERGE (m:Memory {text: mem})
                  MERGE (e)-[:HAS_MEMORY]->(m)
                """,
                {"uuid": message.uuid, "group_id": group_id, "memories": memories}
            )

        print("[Graphiti] Finished memory extraction")

    except Exception as e:
        print(f"[Graphiti] ERROR in extractMemories: {e}")
