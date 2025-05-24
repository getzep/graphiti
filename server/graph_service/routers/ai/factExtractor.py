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
Extract only explicit content facts from the following message—that is, statements about events, actions or situations the speaker describes.
Do NOT extract any meta-information about the message itself (language, style, typos, tone), opinions, emotions, wishes, uncertainties, or ambiguous fragments.
If no such content facts exist, return an empty array.

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
            temperature=TEMPERATURE
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
                    MERGE (e)-[:IS_FACT {group_id: $group_id}]->(f)
                """, {
                    "fact": fact,
                    "uuid": message.uuid,
                    "group_id": group_id
                })

        print(f"[Graphiti] Facts added: {facts}")
        print("[Graphiti] Finished fact extraction")

    except Exception as e:
        print(f"[Graphiti] ERROR in extractFacts: {e}")
