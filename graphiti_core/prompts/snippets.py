summary_instructions = """
        Guidelines:
        1. Output only factual content. Never explain what you're doing, why, or mention limitations/constraints. 
        2. Only use the provided messages, entity, and entity context to set attribute values.
        3. Keep the summary concise and to the point. STATE FACTS DIRECTLY IN UNDER 250 CHARACTERS.

        Example summaries:
        BAD: "This is the only activity in the context. The user listened to this song. No other details were provided to include in this summary."
        GOOD: "User played 'Blue Monday' by New Order (electronic genre) on 2024-12-03 at 14:22 UTC."
        BAD: "Based on the messages provided, the user attended a meeting. This summary focuses on that event as it was the main topic discussed."
        GOOD: "User attended Q3 planning meeting with sales team on March 15."
        BAD: "The context shows John ordered pizza. Due to length constraints, other details are omitted from this summary."
        GOOD: "John ordered pepperoni pizza from Mario's at 7:30 PM, delivered to office."
        """
