# C2 default extraction prompt

# 1. Overview

This change makes the C2 prompt the default text of
`extract_nodes_and_edges.extract_message`. C2 comes from a Zep internal evaluation on GPT-6 Luna.

## 1.1. Evidence

| Test | Current prompt | C2 |
|---|---:|---:|
| Extraction golds, test (455 rows), semantic F1 | 0.628 | 0.638 |
| Extraction golds, train (400 rows), semantic F1 | 0.654 | 0.633 |
| Under-extraction probe, production request shape, runs with a fact | 333/375 (89%) | 360/375 (96%) |

- The noise on the golds is approximately ±0.025 F1. Thus, the golds show no
  clear gain and no clear loss.
- All tests used `gpt-6-luna` with no reasoning. No test used another model.
- The tests used the vendored data sections, not the open-source data
  sections.

# 2. Change

- `_SYSTEM_PROMPT` becomes the C2 system text.
- The instruction text of `_build_extract_user_prompt` becomes the C2
  instruction text. The C2 text replaces the entity rules, the fact rules
  (`_PAIRWISE_FACT_RULES`), and the negative examples.
- The data sections stay the same: `<ENTITY TYPES>`, the fact types section,
  the custom instructions section, `<PREVIOUS MESSAGES>`, and
  `<CURRENT MESSAGES>`.
- The response model `CombinedExtraction` and the prompt name stay the same.
- The exact C2 system and instruction text appears in Appendix A below.

# 3. Blast radius

All Graphiti users that use the combined extraction prompt will get the new
prompt, on all models. A user can keep the old prompt through a prompt
override. The release will be a minor version.

# 4. Tests

- Update the prompt tests that check the current rule text.
- Add a test that the data sections are byte-identical to the current render
  for the same context.
- Add a test that the custom instructions and the fact types still render.

# Appendix A. C2 prompt text

## A.1. System text

```text
You build the long-term memory of an AI agent. From each conversation you extract the entities and the relationship facts that the agent will need later to answer questions, personalize responses, and continue work. The original conversation will not be available later: only your entities and facts will remain, so each fact must stand alone.
```

## A.2. Instruction text

```text
# Goal
Turn CURRENT MESSAGES into a knowledge graph: a list of entities and a list of facts (edges) between them. A good result lets an agent answer "who said, did, wants, owns, plans, feels, recommends, or knows what" without the source text.

# What good output looks like
- Complete: every episode, post, and speaker with real content is covered, including the later ones in a long input. Preferences, opinions, plans, commitments, requests, questions, emotions, experiences, advice, and results all count. Skip only content-free turns ("Hi!", "Thanks!", "Bye!").
- Attributed: when a named person, author, account, or organization states, recommends, or does something, that actor is the source entity, and the fact text names the actor.
- Self-contained: each fact is one plain sentence that uses entity names (no pronouns), keeps the specific details (numbers, dates, prices, versions, names), and makes sense with no other context.
- Compact: one fact for one statement. Keep a list of items from one statement in one fact ("Marina plans to change into a sweater and trousers."), unless each item has its own details. Do not make a fact or an entity for each list item. Not redundant: If the same fact appears in more than one episode, emit it once and list all of its episode indices.

# Entities
- Speakers. Name each speaker with the most complete name available. When a message header has profile fields (first_name and last_name, given_name and family_name, user_alias, screen_name), use the full personal name, for example "Marisol Beckett". Use "User" only when no name is available.
- Other entities: named people, organizations, places, products, files, tickets, projects, events, documents, and named or described objects, plus hobbies and activities that someone does ("road cycling", "watercolor painting"). Use the most specific form mentioned, at most 5 words, as a noun phrase. For a person's possession, pet, project, or creation, use a possessive name ("Calvin's guitar", "Marina's sweater"), not the bare noun.
- Do not make entities from: pronouns or unresolved references ("the city", "this issue"); vague abstractions ("growth", "life", "time"); clock times, dates, quantities, prices, durations, measurements, or coordinates (keep these in the fact text, unless the value is itself an edge endpoint such as a version string); answer options or template tokens ("Agree", "Strongly disagree"); imperative tips ("Buy in bulk" becomes the topic "bulk buying"); quoted slogans or idioms; tutorial example values ("/path/to/source", "file.txt").
- Each entity appears once. Classify it with an entity_type_id from ENTITY TYPES. Extract entities only from CURRENT MESSAGES; use PREVIOUS MESSAGES and [CONTEXT EPISODE] messages only to resolve references.

# Facts
- source_entity_name and target_entity_name must be names from extracted_entities. If an endpoint is a value such as a version, add it as an entity.
- Connect the actor directly to the thing: "Calvin -> PERFORMED_IN -> Tokyo", "Mira Sen -> RECOMMENDS_METRICS_FOR -> Ember & Oak". Do not route a relationship through descriptive nouns ("city lights -> LOCATED_IN -> Tokyo"), and do not turn an actor's claim into a fact between two nouns that the actor mentioned.
- Use a self-referencing fact (source and target are the same entity) only when a statement is about one entity and no other extracted entity fits, for example a routine, a feeling, or a health state ("Evan has a knee injury.").
- When a fact involves two extracted entities, use both as source and target.
- Use PREVIOUS MESSAGES and [CONTEXT EPISODE] messages to resolve what the current message refers to, and write the resolved fact ("all the hard work paid off" becomes what paid off). [CONTEXT EPISODE] messages are conversation turns between the numbered episodes, shown so you can read the conversation in order.
- Set episode_indices to the 0-based [Episode N] numbers that the fact comes from. Messages headed [CONTEXT EPISODE] are not numbered episodes: do not extract facts from them and do not give them an episode index.
- Write each entity name and each fact in the dominant language of its source episode. Keep proper names as written. Write relation_type as an English SCREAMING_SNAKE_CASE label.
- A fact states what the source says. Do not add reasoning, hedges ("appears to", "suggests"), or commentary.

# Output
Return only the JSON object: {"extracted_entities":[{"name":"...","entity_type_id":0}],"edges":[{"source_entity_name":"...","target_entity_name":"...","relation_type":"...","fact":"...","episode_indices":[0]}]}.
```
