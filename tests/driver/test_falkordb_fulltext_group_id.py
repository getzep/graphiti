"""FalkorDB fulltext group filter: group ids must match the way RediSearch tokenized them."""

from graphiti_core.driver.falkordb.fulltext import _group_id_phrase, build_falkor_fulltext_query


def test_plain_group_id_is_quoted_unchanged():
    assert build_falkor_fulltext_query('alice', ['main']) == '(@group_id:"main") (alice)'


def test_hyphenated_group_id_becomes_the_indexed_phrase():
    # RediSearch splits TEXT fields on separators, so the stored value "ko-verify" is the
    # token sequence "ko verify". The escaped form "ko\-verify" never matched anything, which
    # made every BM25 search scoped to a hyphenated group return no results.
    assert build_falkor_fulltext_query('alice', ['ko-verify']) == '(@group_id:"ko verify") (alice)'


def test_multiple_group_ids_are_ored():
    assert (
        build_falkor_fulltext_query('alice', ['main', 'team-a'])
        == '(@group_id:"main"|"team a") (alice)'
    )


def test_non_separator_special_characters_stay_escaped():
    # '_' is not a separator, so it is not a phrase boundary - but an unescaped '_' in the
    # phrase matches nothing, and '_' is FalkorDB's default group id.
    assert build_falkor_fulltext_query('alice', ['team_a']) == '(@group_id:"team\\_a") (alice)'
    assert build_falkor_fulltext_query('alice', ['_']) == '(@group_id:"\\_") (alice)'


def test_group_id_phrase_leaves_non_latin_letters_unescaped():
    # validate_group_id rejects such ids today, but the phrase builder must not escape letters:
    # RediSearch does not match an escaped Hangul syllable against the indexed token.
    assert _group_id_phrase('한국어-그룹') == '"한국어 그룹"'
