"""Claude Haiku 5.5 extraction and timestamp prompt variant."""

import re
from typing import Any

from graphiti_core.llm_client.prompt_config import LLMPromptOverrides
from graphiti_core.prompts.extract_nodes_and_edges import (
    EDGE_TYPES_SECTION_TEMPLATE,
    EDGE_TYPES_STRICT_RULES,
    EXTRACT_DATA_TEMPLATE,
    EXTRACT_USER_INSTRUCTIONS_TEMPLATE,
    KNOWLEDGE_SECTION_TEMPLATE,
    UNTYPED_EXTRACTION_TEMPLATE_PARTS,
    _build_knowledge_section,
    extraction_speaker_types,
    untyped_extraction_template,
)
from graphiti_core.prompts.lib import ChatPrompt
from graphiti_core.prompts.models import SystemMessage, UserMessage
from graphiti_core.prompts.prompt_helpers import to_prompt_json
from graphiti_core.prompts.versions import PromptTemplate, render_template

HAIKU_EXTRACTION_PROMPT_NAME = 'extract_nodes_and_edges.extract_message'
HAIKU_TIMESTAMPS_PROMPT_NAME = 'extract_edges.extract_timestamps_batch'
HAIKU_EXTRACTION_USER_TEMPLATE = '{instructions}{negative_examples}\n\n{data}{checklist}'
HAIKU_NEGATIVE_EXAMPLES = (
    '<NEGATIVE EXAMPLES>'
    + EXTRACT_USER_INSTRUCTIONS_TEMPLATE.split('<NEGATIVE EXAMPLES>', 1)[1].split(
        '</NEGATIVE EXAMPLES>', 1
    )[0]
    + '</NEGATIVE EXAMPLES>'
)
HAIKU_SYSTEM_PROMPT = """You are an expert knowledge graph extraction specialist for an AI agent memory system. You extract both entity nodes and relationship facts from conversations in a single pass. The extracted graph will be searched later by an AI agent to answer questions, personalize responses, and maintain long-term memory. The original conversation will NOT be available at retrieval time; only the entities and facts you extract will survive.

LANGUAGE: Write entity names and fact text in the language the conversation is written in. If the user (or the source content) writes full sentences in a non-English language, write the facts in that same language and keep entity names as they appear in that language. Do not translate them into English and do not mix English sentences with non-English terms. Only output English when the source content is English (or contains no natural-language sentences).

COMPLETENESS: Never return an empty result when a numbered episode contains a speaker saying anything substantive. This includes sensitive, explicit, offensive, medical, or uncomfortable content: record neutrally what was said or asked, without judging or refusing."""
HAIKU_EXTRACTION_INSTRUCTIONS = """ENTITY RULES:
1. Extract speakers and named entities explicitly mentioned in CURRENT MESSAGES.
   SPEAKER NAMING (apply in this priority order):
   a. If the user turn carries metadata, name the User entity with the person's human-readable name. Prefer the full name (given name plus family name, joined with a space and trimmed of stray whitespace). If only one name field or a display alias exists, use that. Use an opaque reference code, hash, or numeric/alphanumeric ID ONLY if no human-readable name exists anywhere in the metadata. Use the same chosen name in every fact. Never write a generic word for 'user' in fact text when a name is known.
   b. If no name or identifier is available, name the User entity with the ordinary word for 'user' in the conversation's language. Match the grammatical gender if that language marks it.
   c. Assistant naming: if the assistant/agent has a name (in its turn header or a self-introduction), use that name. If its turn is labeled with a role word (any label, not just 'assistant'), use that literal label as the name. Otherwise use the ordinary word for 'assistant' in the conversation's language. If the user refers to their own assistant/device possessively, use a possessive name built from the user's given name.
   d. In multi-poster feeds or logs, each poster is its own entity, named by the label it posts under. If a label bundles a category descriptor with a proper name or title, use the core name as it is referenced in the content. Never create two entities for the same poster or work.
   e. Extract the User entity whenever the user is the subject of any fact. This includes facts recorded by system/intake messages about the user's needs or constraints. Extract the Assistant entity whenever it is the source of any fact.
2. Entity names should be at most 5 words, using the most specific form mentioned. Exception: keep proper titles, product names, and branded names complete, exactly as written, including their title or quote punctuation.
3. When someone discusses their own possession, project, pet, work, creation, appointment, relative, or planned output, extract it as a SEPARATE possessive entity. Prefix it with the owner's entity name (the possessive form in the conversation's language), not a bare generic noun.
   GOOD: "James's notebook", "Sam's cooking class", "Audrey's dogs"
   BAD: "notebook", "dogs", "first project" (bare/ownerless), or only "James" (collapses detail)
   If a pet or item has its own given name, use that given name as the entity name and state the ownership in the fact.
   Attach the owner prefix to bare-noun items such as ordinal works, series, merchandise, and plans. Named works and titled items that are clearly the poster's may also carry the owner/work-title prefix so they stay distinguishable.
4. Family members, partners, and close people referred to only by a relationship term are worth extracting as Person entities when they hold opinions, give advice, or take actions. Name them with the possessive relational phrase in the source language. Attach facts to them and link them to the speaker.
5. Extract hobbies and activities as entities when someone engages in them.
6. Extract named/described objects and places, not bare generics. Named places mentioned by ANY speaker are entities. Cities listed as candidates or options are each their own entity. Named services, products, labels, files, and system components count as entities. Care providers or specialists the user will see (e.g., a medical specialty referenced as who the user is seeing) are entities linked directly from the user. Do not wrap them in a generic 'appointment' node.
7. Entity names must be NOUN PHRASES referring to a discrete thing in the speaker's world. Do NOT extract:
   a. Pronouns, articles, unresolved references. Resolve to the referent or skip.
   b. Vague abstractions and filler nouns, or full sentences/propositions as names.
   c. Clock times and dates. Keep them in fact text.
   d. Bare quantities, measurements, durations, prices, counts when only attributes. EXCEPTION: when a value IS an edge endpoint (versions, pinned releases, etc.), extract it as an entity.
   e. Geographic coordinates.
   f. Multiple-choice labels, response-option tokens, template scaffolding.
   g. Imperative advice phrases from tip lists.
   h. Quoted slogans, idioms, loaded adjective phrases.
   i. A speaker's own feelings, bodily symptoms, medical findings, self-described traits/styles, or the abstract subject of their request (e.g., 'strategies for X'). Record these as self-referencing facts on the speaker, or as edges from the advisor to the speaker. Do not create concept nodes for them.
8. Each entity appears exactly ONCE. Classify using the ENTITY TYPES provided. The User entity uses the User type; the Assistant uses the Assistant type.
9. Only extract entities from CURRENT MESSAGES. PREVIOUS MESSAGES and [CONTEXT EPISODE] messages are context only.
10. Skip didactic/tutorial scaffolding when the assistant teaches a topic. The user's interest in the topic is one fact.
11. Every extracted entity must participate in at least one edge. Prefer FEWER, well-anchored entities: do not create an entity just to hold a description that fits inside a fact.

FACT RULES:
1. source_entity_name and target_entity_name MUST exactly match names in extracted_entities.
2. When a fact involves two extracted entities, use both as source and target.
   - A speaker reporting the state of their own possession: speaker -> ... -> possession.
   - A speaker reporting their OWN feelings, symptoms, traits, states, or requests for help: use a self-referencing fact (speaker -> ... -> speaker). Do NOT route it to the listener/assistant merely because they were addressed.
   - An assistant/responder giving advice, explanations, suggestions, or offers to the user: assistant -> ... -> User, with the advice content in the fact text. Do not route it to an abstract topic node.
   - Reported speech from an unnamed third party (e.g., 'she said'): attribute it to the user as something the user was told. Never attribute it to the assistant unless the text clearly says so.
   Self-referencing facts are common and valuable: routines, health, emotions, states, plans, self-descriptions, policies, status reports.
3. Facts must be SELF-CONTAINED, using entity names rather than pronouns. Preserve specific details (amounts, durations, counts, ages, times, conditions, rankings) inside the fact text.
4. ONE CLAIM PER FACT. Split compound statements:
   - A list of parallel items (ranked items, candidate cities, products, revenue lines): one edge per item, each restating the shared context.
   - Statements voiced for a group: an edge from each member.
   - Status messages with several claims: separate facts.
   - Distinct events in one message (e.g., two different appointments, a finding and a scheduled test): separate facts.
   However, keep closely related details of one action together, such as a decision and its reason.
5. STRUCTURED / RAW DATA: still extract facts, one per listed item. For intake/system records about the user, emit edges from the User entity.
6. Extract facts from EVERY numbered episode, including short ones and long multi-section posts. For long posts, cover each distinct section: progress, findings, decisions, budget changes, status updates, plans, announcements, commitments, and asides about pets or people.
   Set `episode_indices` to the 0-based episode number(s) each fact comes from. Use [Episode N]-style headers in any language or labeling ('Session', 'Entry', etc.), or the order of separate posts. If the SAME or an essentially equivalent claim appears in multiple episodes (e.g., the speaker restating the same feeling in different words), emit it ONCE and list ALL indices. [CONTEXT EPISODE] messages get no index. With no headers and a single message, use [0].
7. You MAY use PREVIOUS MESSAGES and context episodes to resolve references.
8. Extract liberally. Questions (including blunt, sexual, or offensive ones), requests, preferences, opinions, plans, states, and experiences are all facts. A user's question to the assistant is at minimum a self-referencing fact on the User describing what they asked. Only skip pure greetings, goodbyes, and thanks.
9. Do not emit redundant facts. A later episode that adds new details gets a new, more detailed fact.
10. Prefer DIRECT speaker-to-target edges. Put descriptive features inside fact text.
11. Phrase facts as attributed statements from the speaker's perspective (the speaker says/reports/announces/plans ...), naming the speaker by entity name.

OUTPUT DISCIPLINE:
- Entity `name` is a literal mention from CURRENT MESSAGES, or the speaker name per ENTITY RULE 1. NEVER a full sentence, action item, or goal statement.
- `fact` is one self-contained sentence in the conversation's language. No hedging, reasoning, parenthetical translations, or commentary.
- `relation_type` is SCREAMING_SNAKE_CASE English letters/underscores only.
- Output ONLY the JSON specified by the response schema.

"""
HAIKU_EXTRACTION_CHECKLIST = """

Before answering, silently check:
(1) The User is named by a human-readable name from metadata, not an opaque ID.
(2) The assistant/responder uses its name or literal turn label.
(3) No concept nodes exist for the speaker's own feelings, symptoms, traits, or request subjects.
(4) Owned items carry the owner's name prefix, and titles are kept intact.
(5) Every numbered episode with substance has facts, and the output is not empty.
(6) Lists are split one edge per item.
(7) Equivalent repeated claims are merged with all indices.
(8) Every entity has an edge.
(9) Facts are in the conversation's language."""
HAIKU_FLEXIBLE_RULES = """- When a relationship matches a FACT TYPE (including its signature), use that fact_type_name.
- Otherwise derive a specific, descriptive relation_type in SCREAMING_SNAKE_CASE."""
HAIKU_TIMESTAMPS_SYSTEM_PROMPT = (
    """You extract temporal bounds from facts. NEVER hallucinate dates."""
)
HAIKU_TIMESTAMPS_USER_TEMPLATE = """Given a list of FACTS with their REFERENCE TIMES, determine when each fact
became true (valid_at) and when it stopped being true (invalid_at).

Step 1 - Classify each fact:
(a) Ongoing state in present tense (e.g., "X works as Y", "X lives in Z", "X serves as ..."): valid_at = REFERENCE TIME exactly; invalid_at null unless an end is stated.
(b) Past-tense one-off action or event with NO explicit or relative time expression in the text (e.g., "X sent a message", "X opened the file", "X delivered the package"): valid_at = null AND invalid_at = null. Do NOT use the REFERENCE TIME as a fallback for completed past actions; the reference time is when the fact was recorded, not when the event happened.
(c) Fact containing an explicit date or a relative time expression: resolve it as described below.
(d) Future-tense plan, promise, intention, scheduled event, or deadline: valid_at = REFERENCE TIME; do NOT set valid_at to the future occurrence date.

Rules for resolving time expressions (relative to each fact's own REFERENCE TIME):
- Day-granularity relative expressions ("yesterday", "N days ago", "in N days", "today" when describing a past event): compute the calendar date and use T00:00:00Z (midnight). Do NOT carry over the reference time-of-day. E.g., reference 2019-09-15T18:40:10.123456Z, "did X 3 days ago" -> valid_at 2019-09-12T00:00:00Z, invalid_at null.
- Calendar-period expressions describing a past period during which something happened ("last week", "last month", "last year"): set valid_at to the start of that period and invalid_at to the start of the following period, both at T00:00:00Z. Weeks run Monday to Monday (ISO weeks): "last week" = Monday 00:00 of the previous week through Monday 00:00 of the current week. "last month" = 1st of previous month through 1st of current month. "last year" = Jan 1 of previous year through Jan 1 of current year. E.g., reference Thursday 2019-09-12T08:00:00Z, "did X last week" -> valid_at 2019-09-02T00:00:00Z, invalid_at 2019-09-09T00:00:00Z.
- "N years/months/weeks ago": subtract from the reference date, use T00:00:00Z.
- Preserve future occurrence dates and deadlines in the fact text. Do NOT use a soft deadline ("by March 2026", "by the end of Q3 2025") as invalid_at.
- Use a future date as invalid_at when the fact describes an ongoing state that ends then (e.g., "is unavailable until December 25, 2024", "is paused until June 1").
- Bounded durations: if a state is stated to last a fixed period from now (e.g., "has 14 days of access", "remains valid for 30 days", "is active for two weeks"), set valid_at = REFERENCE TIME and invalid_at = REFERENCE TIME + that duration, keeping the exact time-of-day and fractional seconds of the REFERENCE TIME.
- Dates without a year: for a future end, deadline, or scheduled date (e.g., "until March 3", "will meet on March 3"), use the next occurrence of that month/day on or after the REFERENCE TIME. For a past event or completed action, use the most recent occurrence on or before the REFERENCE TIME.
- If a change or end is expressed, set invalid_at to the relevant time.
- Leave both null if no time is stated or resolvable and the fact is not a present-tense ongoing state.
- If only a date is mentioned (no time), use 00:00:00.

Timestamp format (critical):
- Output ISO 8601 with a Z suffix.
- When a value IS the REFERENCE TIME (ongoing states, future plans, bounded-duration starts/ends), copy it EXACTLY including ALL fractional-second digits; only replace "+00:00" with "Z". Never truncate or round. Example: "2021-05-06T07:08:09.123456+00:00" -> "2021-05-06T07:08:09.123456Z".
- Dates from the fact text or from day/week/month/year-level relative expressions use "T00:00:00Z".
- Do NOT hallucinate or infer dates from unrelated events.

Examples with REFERENCE_TIME 2020-01-01T00:00:00Z:
  - "Emma will relocate to Tokyo on December 25, 2022" -> valid_at 2020-01-01T00:00:00Z, invalid_at null.
  - "The team will ship version 2.0 by the end of Q3 2025" -> valid_at 2020-01-01T00:00:00Z, invalid_at null.
Examples with REFERENCE_TIME 2020-03-10T12:30:45.500000+00:00 (a Tuesday):
  - "Dana manages the support team" -> valid_at 2020-03-10T12:30:45.500000Z, invalid_at null.
  - "Dana forwarded the invoice" -> valid_at null, invalid_at null.
  - "Dana visited Lisbon on February 2" -> valid_at 2020-02-02T00:00:00Z, invalid_at null.
  - "Dana fixed the bug yesterday" -> valid_at 2020-03-09T00:00:00Z, invalid_at null.
  - "Dana logged 40 hours last week" -> valid_at 2020-03-02T00:00:00Z, invalid_at 2020-03-09T00:00:00Z.
  - "The trial account is active for 10 days" -> valid_at 2020-03-10T12:30:45.500000Z, invalid_at 2020-03-20T12:30:45.500000Z.
  - "The store is closed until February 2" -> valid_at 2020-03-10T12:30:45.500000Z, invalid_at 2021-02-02T00:00:00Z.

Return exactly one timestamps entry per fact, in the same order (including duplicate facts).

<FACTS>{facts}</FACTS>"""


def is_haiku_model(name: str) -> bool:
    """Recognize direct Anthropic and Bedrock Haiku 5.5 model ids."""
    return (
        re.fullmatch(
            r'(?:anthropic/)?(?:(?:(?:us|eu|global)\.)?anthropic\.)?'
            r'claude-haiku-5-5(?:-[a-zA-Z0-9][a-zA-Z0-9.:-]*)?',
            name,
        )
        is not None
    )


def haiku_extract_message(context: dict[str, Any]) -> ChatPrompt:
    edge_types = context.get('edge_types')
    edge_section = ''
    if edge_types:
        edge_section = render_template(
            EDGE_TYPES_SECTION_TEMPLATE,
            edge_types=to_prompt_json(edge_types),
            relation_type_rules=(
                EDGE_TYPES_STRICT_RULES
                if context.get('strict_edge_types', False)
                else HAIKU_FLEXIBLE_RULES
            ),
        )
    data_template = EXTRACT_DATA_TEMPLATE
    instructions = HAIKU_EXTRACTION_INSTRUCTIONS
    if context.get('untyped_entities'):
        data_template = untyped_extraction_template(data_template)
        instructions = untyped_extraction_template(instructions, extraction_speaker_types(context))
    data = render_template(
        data_template,
        entity_types=context['entity_types'],
        edge_types_section=edge_section,
        knowledge_section=_build_knowledge_section(context.get('knowledge')),
        previous_episodes=to_prompt_json(list(context['previous_episodes'])),
        episode_content=context['episode_content'],
    )
    content = render_template(
        HAIKU_EXTRACTION_USER_TEMPLATE,
        instructions=instructions,
        negative_examples=HAIKU_NEGATIVE_EXAMPLES,
        data=data,
        checklist=HAIKU_EXTRACTION_CHECKLIST,
    )
    return ChatPrompt(
        system=SystemMessage(content=HAIKU_SYSTEM_PROMPT),
        user=UserMessage(content=content),
    )


def haiku_extract_timestamps_batch(context: dict[str, Any]) -> ChatPrompt:
    return ChatPrompt(
        system=SystemMessage(content=HAIKU_TIMESTAMPS_SYSTEM_PROMPT),
        user=UserMessage(
            content=render_template(
                HAIKU_TIMESTAMPS_USER_TEMPLATE, facts=to_prompt_json(context['facts'])
            )
        ),
    )


haiku_extract_message.__prompt_template__ = PromptTemplate(
    (
        HAIKU_SYSTEM_PROMPT,
        HAIKU_EXTRACTION_INSTRUCTIONS,
        HAIKU_EXTRACTION_CHECKLIST,
        HAIKU_FLEXIBLE_RULES,
        HAIKU_EXTRACTION_USER_TEMPLATE,
        HAIKU_NEGATIVE_EXAMPLES,
        EXTRACT_DATA_TEMPLATE,
        EDGE_TYPES_SECTION_TEMPLATE,
        EDGE_TYPES_STRICT_RULES,
        KNOWLEDGE_SECTION_TEMPLATE,
        *UNTYPED_EXTRACTION_TEMPLATE_PARTS,
    )
)
haiku_extract_timestamps_batch.__prompt_template__ = PromptTemplate(
    (
        HAIKU_TIMESTAMPS_SYSTEM_PROMPT,
        HAIKU_TIMESTAMPS_USER_TEMPLATE,
    )
)


def haiku_prompt_overrides() -> LLMPromptOverrides:
    return LLMPromptOverrides(
        extract_nodes_and_edges=LLMPromptOverrides.ExtractNodesAndEdges(
            extract_message=haiku_extract_message,
        ),
        extract_edges=LLMPromptOverrides.ExtractEdges(
            extract_timestamps_batch=haiku_extract_timestamps_batch,
        ),
    )
