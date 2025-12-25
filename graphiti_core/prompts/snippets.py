"""
Copyright 2024, Zep Software, Inc.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

summary_instructions = """Guidelines:
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
