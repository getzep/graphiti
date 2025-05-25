"""
Function specifications for OpenAI API calls.
"""

# Define OpenAI function specifications for facts, emotions, and entities
functionsSpec = [
   {
        "name": "extractFacts",
        "description": "List of objective facts based on observable events or actions. Each fact must be concise and exclude feelings or thoughts.",
        "parameters": {
            "type": "object",
            "properties": {
                "facts": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "List of short factual statements"
                }
            },
            "required": ["facts"]
        }
    },
    {
        "name": "extractEmotions",
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
    },
   {
        "name": "extractEntities",
        "description": "Identify specific people mentioned in the message.",
        "parameters": {
            "type": "object",
            "properties": {
                "entities": {
                    "type": "array",
                    "items": { "type": "string" },
                    "description": "List of specific people mentioned in the message. Include names or clear roles (e.g. 'my neighbor', 'dad', 'Marta'). Do not include vague pronouns like 'she', 'someone', 'they'."
                }
            },
            "required": ["entities"]
        }
    }
]
