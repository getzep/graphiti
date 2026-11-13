from graphiti_core.search.search_utils import DEFAULT_RRF_RANK_CONST, rrf


def test_default_constant_is_the_standard_sixty():
    assert DEFAULT_RRF_RANK_CONST == 60


def test_agreement_across_lists_beats_a_single_top_hit():
    # X is ranked 4th by both retrievers; Y is ranked 1st by one of them only.
    # RRF's purpose is that cross-list agreement wins - with rank_const=1 it did not.
    ranked, scores = rrf([['Y', 'p', 'q', 'X'], ['r', 's', 't', 'X']])
    assert ranked[0] == 'X'
    assert scores[0] > scores[1]


def test_rank_const_one_reproduces_the_old_winner_take_all_behaviour():
    ranked, _ = rrf([['Y', 'p', 'q', 'X'], ['r', 's', 't', 'X']], rank_const=1)
    assert ranked[0] != 'X'


def test_min_score_filters_by_fused_score():
    ranked, scores = rrf([['a', 'b'], ['a', 'c']], min_score=2 / 61)
    assert ranked == ['a']
    assert len(scores) == 1
