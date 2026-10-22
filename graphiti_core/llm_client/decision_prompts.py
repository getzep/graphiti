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

import json
import re
from typing import Any

from graphiti_core.prompts.lib import (
    ChatPromptLibrary,
    default_chat_prompt_library,
    get_prompt_builder,
)
from graphiti_core.prompts.models import ChatPrompt, Message, SystemMessage, UserMessage

from .prompt_config import LLMPromptOverrides

RULES_HEADER = '\n\nRULES FOR THE QUESTIONS:\n'

NODE_RULES = (
    'Entities are duplicates only if they refer to the same real-world object or concept. '
    'They are NOT duplicates if they are related but distinct, or if they have similar names or '
    'purposes but refer to separate instances or concepts. The same name for different real-world '
    'things (for example Java the programming language and Java the island) is not a duplicate. '
    'An abbreviation, a synonym, or a more complete name for the same thing with the same '
    "possessor is a duplicate (for example NYC and New York City; Marco's car and Marco's vehicle)."
)

NODE_NONE = (
    'No EXISTING ENTITY refers to the same real-world object or concept as this entity. '
    'Related but distinct entities, separate instances, and entities that only share a name or '
    'purpose are not duplicates.'
)

DUP_RULES = (
    'A duplicate states identical factual information: the same assertion about the same '
    'entities, also when the wording or the language is different. An existing fact is also a '
    'duplicate when it already states all of the information in the NEW FACT, even if the '
    'existing fact has more detail (for example, extra identifiers). It is NOT a duplicate when '
    'the NEW FACT has a different numeric value, date, identifier, or key qualifier, when the '
    'NEW FACT adds information that the existing fact does not state, or when the facts '
    'describe different events.'
)

CON_RULES = (
    'A contradiction means that the NEW FACT and the old fact cannot both be true. It needs '
    'explicit evidence that the old fact stopped being true or was replaced, such as termination, '
    'replacement, transfer, or mutually exclusive current-state language (for example a changed '
    'job title for the same job). These are NOT contradictions: a restatement, paraphrase, or '
    'translation of the same fact; a repeated event; different events on different dates (Bob ran '
    '5 miles on Tuesday and Bob ran 3 miles on Wednesday); recency alone; new activity, progress, '
    'status, updates, blockers, or collaboration about the same person or project, which do not '
    'contradict a role, ownership, responsibility, membership, or assignment fact; and a second '
    'employer, role, or location that can be true at the same time.'
)


def tag(text, name):
    m = re.search(rf'<{name}>\s*(.*?)\s*</{name}>', text, re.S)
    return m.group(1) if m else None


def parse_list(text: str | None) -> list[dict[str, Any]]:
    if text is None or not text.strip():
        return []
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    import ast

    try:
        return ast.literal_eval(text)
    except (SyntaxError, ValueError):
        pass
    out: list[dict[str, Any]] = []
    for m in re.finditer(r"\{.*?\}(?=\s*'?,|\s*'?\]$|\s*'?\]\s*$)", text, re.S):
        span = m.group(0)
        obj: Any = None
        for loader in (json.loads, ast.literal_eval):
            try:
                obj = loader(span)
                break
            except Exception:
                continue
        if not isinstance(obj, dict):
            obj = {'raw': span}
            for key in ('candidate_id', 'idx', 'id'):
                km = re.search(rf'["\']{key}["\']\s*:\s*(-?\d+)', span)
                if km:
                    obj[key] = int(km.group(1))
            for key in ('name', 'fact'):
                km = re.search(rf'["\']{key}["\']\s*:\s*["\'](.*?)["\']\s*[,}}]', span, re.S)
                if km:
                    obj[key] = km.group(1)
        out.append(obj)
    return out


def shrink_state(state):
    """Drop PREVIOUS MESSAGES first, then halve the CURRENT MESSAGE from the front."""
    prev = tag(state, 'PREVIOUS MESSAGES')
    if prev and prev.strip() not in ('[]', ''):
        return state.replace(prev, '[]', 1)
    cur = tag(state, 'CURRENT MESSAGE')
    if cur and len(cur) > 2000:
        return state.replace(cur, '...' + cur[len(cur) // 2 :], 1)
    return None


def node_n6(user):
    """Decompose a rendered dedupe_nodes.nodes user message into state + questions."""
    entities = parse_list(tag(user, 'ENTITIES'))
    existing = parse_list(tag(user, 'EXISTING ENTITIES'))
    state = user.split('</EXISTING ENTITIES>')[0] + '</EXISTING ENTITIES>'
    criteria = {
        str(c['candidate_id']): f'candidate_id {c["candidate_id"]}: {c.get("name")}'
        for c in existing
        if 'candidate_id' in c
    }
    criteria['none'] = NODE_NONE
    questions = {
        f'e{e["id"]}': {
            'type': 'choice',
            'instructions': (
                f'The entity with id {e["id"]} in ENTITIES is "{e.get("name")}" '
                f'({", ".join(e.get("entity_type") or [])}). It was extracted from the CURRENT MESSAGE. '
                'Which EXISTING ENTITY (by candidate_id) refers to the same real-world object or concept '
                'as this entity? Obey the RULES. Choose none if there is no duplicate.'
            ),
            'criteria': criteria,
        }
        for e in entities
    }
    return state + RULES_HEADER + NODE_RULES, questions, entities, existing


def _contra_strict(new_fact, f):
    return {
        'type': 'noul',
        'instructions': (
            f'Does the NEW FACT "{new_fact}" directly contradict the fact idx={f["idx"]} '
            f'"{f.get("fact", f.get("raw", ""))}", so that both cannot be true at the same time? A contradiction '
            'needs explicit evidence that the old fact stopped being true or was replaced '
            '(termination, replacement, transfer, or mutually exclusive current-state '
            'language). New activity, progress, or status about the same entities does not '
            'contradict a role, ownership, or membership fact.'
        ),
        'criteria': {
            'true': 'The old fact and the NEW FACT cannot both be true.',
            'false': 'Both facts can be true, or the NEW FACT is a different event or a duplicate.',
        },
    }


def edge_e2(user):
    """Decompose a rendered dedupe_edges.resolve_edge user message (strict contradiction)."""
    existing = parse_list(tag(user, 'EXISTING FACTS'))
    candidates = parse_list(tag(user, 'FACT INVALIDATION CANDIDATES'))
    new_fact = (tag(user, 'NEW FACT') or '').strip()
    lines = [f'NEW FACT: {new_fact}', '', 'EXISTING FACTS:']
    lines += [f'  idx={f.get("idx")}: {f.get("fact", f.get("raw", ""))}' for f in existing] or [
        '  (none)'
    ]
    lines += ['', 'FACT INVALIDATION CANDIDATES:']
    lines += [f'  idx={f.get("idx")}: {f.get("fact", f.get("raw", ""))}' for f in candidates] or [
        '  (none)'
    ]
    state = '\n'.join(lines)
    questions = {}
    for f in existing:
        questions[f'dup{f["idx"]}'] = {
            'type': 'noul',
            'instructions': (
                f'Does the existing fact idx={f["idx"]} "{f.get("fact", f.get("raw", ""))}" '
                f'state the same information as the NEW FACT "{new_fact}", or already state '
                'all of the information in the NEW FACT? The wording or the language can be '
                'different, and the existing fact can have more detail. It is not a duplicate '
                'when the NEW FACT has a different numeric value, date, identifier, or '
                'qualifier, or information that the existing fact does not state.'
            ),
            'criteria': {
                'true': (
                    'The existing fact states the same information as the NEW FACT, or all of '
                    'it, about the same entities.'
                ),
                'false': (
                    'The NEW FACT has information that the existing fact does not state, a '
                    'detail is different, or the facts describe different events or '
                    'relationships.'
                ),
            },
        }
    for f in existing + candidates:
        questions[f'con{f["idx"]}'] = _contra_strict(new_fact, f)
    return (
        state + RULES_HEADER + 'DUPLICATE: ' + DUP_RULES + '\nCONTRADICTION: ' + CON_RULES,
        questions,
        existing,
        candidates,
    )


def _user_text(messages: list[Message]) -> str:
    for message in messages:
        if message.role == 'user':
            return message.content
    raise ValueError('rendered prompt has no user message')


def _encode_payload(state: str, questions: dict[str, dict[str, Any]]) -> str:
    return json.dumps(
        {'instructions': '', 'state': state, 'questions': questions},
        ensure_ascii=False,
    )


def _jev_node_builder(default_builder):
    """Render the default prompt and convert it to n6 questions."""

    def _builder(context: dict[str, Any]) -> ChatPrompt:
        messages = default_builder(context).as_messages()
        state, questions, _, _ = node_n6(_user_text(messages))
        return ChatPrompt(
            system=SystemMessage(
                content=messages[0].content if messages[0].role == 'system' else ''
            ),
            user=UserMessage(content=_encode_payload(state, questions)),
        )

    return _builder


def _jev_edge_builder(default_builder):
    """Render the default prompt and convert it to e2 questions."""

    def _builder(context: dict[str, Any]) -> ChatPrompt:
        messages = default_builder(context).as_messages()
        state, questions, _, _ = edge_e2(_user_text(messages))
        return ChatPrompt(
            system=SystemMessage(
                content=messages[0].content if messages[0].role == 'system' else ''
            ),
            user=UserMessage(content=_encode_payload(state, questions)),
        )

    return _builder


def assemble_node_answers(
    answers: dict[str, Any], entities: list[dict[str, Any]]
) -> dict[str, Any]:
    """Map Jev choice answers onto NodeResolutions."""
    resolutions = []
    for entity in entities:
        answer = answers.get(f'e{entity["id"]}') or {}
        probabilities = answer.get('probabilities') or {}
        choice = (
            max(probabilities.items(), key=lambda item: item[1])[0]
            if probabilities
            else answer.get('choice')
        )
        candidate_id = -1 if choice in (None, 'none') else int(choice)
        resolutions.append(
            {
                'id': entity['id'],
                'name': entity.get('name'),
                'duplicate_candidate_id': candidate_id,
            }
        )
    return {'entity_resolutions': resolutions}


def assemble_edge_answers(
    answers: dict[str, Any],
    existing_indices: list[int],
    contradiction_indices: list[int],
    duplicate_threshold: float,
    contradiction_threshold: float,
) -> dict[str, Any]:
    """Map Jev noul answers onto EdgeDuplicate."""
    return {
        'duplicate_facts': [
            idx
            for idx in existing_indices
            if (answers.get(f'dup{idx}') or {}).get('noul', 0.0) >= duplicate_threshold
        ],
        'contradicted_facts': [
            idx
            for idx in contradiction_indices
            if (answers.get(f'con{idx}') or {}).get('noul', 0.0) >= contradiction_threshold
        ],
    }


def jev_prompt_overrides(
    library: ChatPromptLibrary | None = None,
) -> LLMPromptOverrides:
    """Return prompt overrides for the Jev n6 and e2 dedupe questions."""
    prompt_library = library if library is not None else default_chat_prompt_library
    return LLMPromptOverrides(
        dedupe_nodes=LLMPromptOverrides.DedupeNodes(
            nodes=_jev_node_builder(get_prompt_builder(prompt_library, 'dedupe_nodes.nodes'))
        ),
        dedupe_edges=LLMPromptOverrides.DedupeEdges(
            resolve_edge=_jev_edge_builder(
                get_prompt_builder(prompt_library, 'dedupe_edges.resolve_edge')
            )
        ),
    )
