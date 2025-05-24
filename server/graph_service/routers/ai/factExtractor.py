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
        "name": "extract_facts",
        "description": "Extract factual statements from the provided message.",
        "parameters": {
            "type": "object",
            "properties": {
                "analysis": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "List of short factual statements"
                }
            },
            "required": ["analysis"]
        }
    }
]

async def extractFactsAndStore(graphiti, message, group_id):
    prompt = f'''\
Extract clear, specific factual statements from the following message.
Return JSON with a single key "analysis" whose value is an array of short facts (no commentary, no emotions).

Message:
"""{message.content}"""
'''

    try:
        print("[Graphiti] Starting fact extraction")
         # Use function calling to get structured facts
        response = openai.chat.completions.create(
            model=OPENAI_MODEL,
            messages=[{"role": "user", "content": prompt}],
            functions=functions_spec,
            function_call="auto",
            temperature=0.3
        )
        # Get the function call arguments from the response
        choice = response.choices[0]
        msg = choice.message
        func_call = msg.function_call
        args = json.loads(func_call.arguments)
        facts = args.get("analysis", [])

        for fact in facts:
            async with graphiti.driver.session() as session:
                await session.run("""
                    MERGE (f:Fact {text: $fact})
                    WITH f
                    MATCH (e:Episodic {uuid: $uuid})
                    WHERE e.group_id = $group_id
                    MERGE (e)-[:IS_FACT]->(f)
                """, {
                    "fact": fact,
                    "uuid": message.uuid,
                    "group_id": group_id
                })

        print(f"[Graphiti] Facts added: {facts}")
        print("[Graphiti] Finished fact extraction")

    except Exception as e:
        print(f"[Graphiti] ERROR in extractFacts: {e}")
