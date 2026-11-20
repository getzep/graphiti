"""Deterministic prompt-contract coverage for pronoun-referenced previous entities.

The separate node-extraction prompt must keep a previous entity that the current
message refers to through a pronoun or another resolvable reference, while still
dropping previous entities the current message does not reference. The combined
node-and-edge prompt already resolves references upstream and is intentionally not
covered here.
"""

from graphiti_core.prompts.extract_nodes import extract_message


def _user_prompt() -> str:
    context = {
        'entity_types': [{'entity_type_id': 0, 'entity_type_name': 'Entity'}],
        'previous_episodes': [
            {'content': "User: This is my friend John. He's a software engineer."}
        ],
        'episode_content': 'User: He also has a friend named Mike.',
        'custom_extraction_instructions': '',
    }
    return ' '.join(extract_message(context)[1].content.lower().split())


def test_separate_extract_message_keeps_pronoun_referenced_previous_entity():
    user_prompt = _user_prompt()

    # A resolved pronoun reference counts as an implicit mention and must be kept.
    assert 'counts as implicitly mentioned' in user_prompt
    assert 'must be extracted' in user_prompt
    assert 'explicitly or implicitly' in user_prompt


def test_separate_extract_message_still_excludes_unreferenced_previous_entities():
    user_prompt = _user_prompt()

    # Previous entities the current message does not reference remain excluded.
    assert 'current message does not refer to them' in user_prompt
