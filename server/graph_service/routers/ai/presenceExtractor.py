import openai
import os
import json

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENAI_MODEL = "gpt-4.1-mini"  # or "gpt-4", adjustable as needed

openai.api_key = OPENAI_API_KEY

# Define function spec for OpenAI function calling
functions_spec = [
    {
        "name": "extract_presence",
        "description": "Detect the presence of factual statements, emotional tones, recollections or memories, and relationship statements in the provided message.",
        "parameters": {
            "type": "object",
            "properties": {
                "fact": {"type": "boolean", "description": "True if any factual statements are present"},
                "emotion": {"type": "boolean", "description": "True if any emotional tones are present"},
                "memory": {"type": "boolean", "description": "True if any memories are mentioned"},
                "relation": {"type": "boolean", "description": "True if any relationships are present"}
            },
            "required": ["fact", "emotion", "memory", "relation"]
        }
    }
]

async def extractPresenceAndStore(graphiti, message, group_id):
    prompt = f'''
Determine whether the following message contains factual statements, emotional tones, recollections or memories, and relationship statements.
Return JSON with keys "fact", "emotion", "memory", "relation" set to true or false accordingly (no commentary).

Message:
"""{message.content}"""
'''

    try:
        print("[Graphiti] Starting presence extraction")
        response = openai.chat.completions.create(
            model=OPENAI_MODEL,
            messages=[{"role": "user", "content": prompt}],
            functions=functions_spec,
            function_call="auto",
            temperature=0.0
        )
        choice = response.choices[0]
        func_call = getattr(choice.message, "function_call", None)
        if func_call and hasattr(func_call, "arguments"):
            data = json.loads(func_call.arguments)
        else:
            data = {}
        fact = data.get("fact", False)
        emotion = data.get("emotion", False)
        memory = data.get("memory", False)
        relation = data.get("relation", False)

        async with graphiti.driver.session() as session:
            await session.run(
                """
                MATCH (e:Episodic {uuid: $uuid})
                WHERE e.group_id = $group_id
                SET e.has_fact = $fact, e.has_emotion = $emotion, e.has_memory = $memory, e.has_relation = $relation
                """,
                {"uuid": message.uuid, "group_id": group_id, "fact": fact, "emotion": emotion, "memory": memory, "relation": relation}
            )
        print("[Graphiti] Presence extraction done", data)
        return data
    except Exception as e:
        print(f"[Graphiti] ERROR in extractPresence: {e}")
