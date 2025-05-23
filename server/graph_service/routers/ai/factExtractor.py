from graph_service.config import get_settings
import openai
import os

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENAI_MODEL = "gpt-4.1-mini"  # or "gpt-4", adjustable as needed

openai.api_key = OPENAI_API_KEY

async def extractFactsAndStore(graphiti, message, group_id):
    prompt = f"""
Extract clear, specific factual statements from the following message.
Return only a list of short facts (no commentary, no emotions).

Message:
\"\"\"{message.content}\"\"\"
"""

    try:
        response = openai.chat.completions.create(
            model=OPENAI_MODEL,
            messages=[
                {"role": "user", "content": prompt}
            ],
            temperature=0.3
        )
        facts_text = response.choices[0].message.content.strip()
        facts = [f.strip("-• ") for f in facts_text.split("\n") if f.strip()]

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

    except Exception as e:
        print(f"[Graphiti] ERROR in extractFacts: {e}")
