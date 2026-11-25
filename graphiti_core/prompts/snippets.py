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

from graphiti_core.utils.text_utils import MAX_SUMMARY_CHARS

summary_instructions = f"""Guidelines:
        1. Output only factual content. Never explain what you're doing, why, or mention limitations or constraints.
        2. Only use the provided messages, entity, and entity context to set attribute values.
        3. Keep the summary information-dense and entity-specific. STATE FACTS DIRECTLY IN UNDER {MAX_SUMMARY_CHARS} CHARACTERS.
        4. Preserve all materially relevant names, roles, places, dates, counts, and temporal qualifiers that are explicitly supported.
        5. Prefer compact factual sentences over vague thematic phrasing or meta-language.
        6. When the durable fact is the content of what was said, state the content directly instead of narrating that it was said.
        7. Use communication verbs only when the act of speaking, asking, sharing, presenting, announcing, or telling is itself the important fact.
        8. Never use filler verbs like "mentioned", "described", "stated", "reported", "noted", "discussed", "referenced", or "indicated" unless the communication act itself is the fact.
        9. Include temporal anchors when the messages provide them and they help ground the fact.
        10. Begin with the entity name or a direct fact, not with "A", "An", "The", or "This is" unless that wording is part of the entity name.

        Example summary:
        BAD: "The context shows John ordered pizza. Due to length constraints, other details are omitted from this summary."
        GOOD: "John ordered pepperoni pizza from Mario's at 7:30 PM and had it delivered to the office."
        """
