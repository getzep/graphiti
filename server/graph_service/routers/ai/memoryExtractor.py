from graph_service.config import get_settings
import openai
import os
import json

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENAI_MODEL = "gpt-4.1-mini"  # or "gpt-4", adjustable as needed

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
Extract clear, specific recollections or memories from the following message.
Return JSON with a single key "memories" whose value is an array of short memory statements (no commentary).

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
            temperature=0.3
        )
        choice = response.choices[0]
        msg = choice.message
        func_call = msg.function_call
        args = json.loads(func_call.arguments)
        memories = args.get("memories", [])

        for memory in memories:
            async with graphiti.driver.session() as session:
                await session.run("""
                    MERGE (m:Memory {text: $memory})
                    WITH m
                    MATCH (e:Episodic {uuid: $uuid})
                    WHERE e.group_id = $group_id
                    MERGE (e)-[:HAS_MEMORY]->(m)
                """, {
                    "memory": memory,
                    "uuid": message.uuid,
                    "group_id": group_id
                })

        print(f"[Graphiti] Memories added: {memories}")
        print("[Graphiti] Finished memory extraction")

    except Exception as e:
        print(f"[Graphiti] ERROR in extractMemories: {e}")
