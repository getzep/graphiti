import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from config.schema import GraphitiConfig, SearchTuningConfig


def test_search_defaults():
    cfg = GraphitiConfig()
    assert isinstance(cfg.search, SearchTuningConfig)
    assert cfg.search.reranker == 'mmr'
    assert cfg.search.mmr_lambda == 0.5
    assert cfg.search.max_facts == 6
    assert cfg.search.max_nodes == 6
    assert cfg.search.exclude_invalidated is True
    assert cfg.search.reranker_min_score == 0.0


def test_search_env_overrides():
    os.environ['SEARCH__RERANKER'] = 'rrf'
    os.environ['SEARCH__MMR_LAMBDA'] = '0.7'
    os.environ['SEARCH__MAX_FACTS'] = '4'
    os.environ['SEARCH__EXCLUDE_INVALIDATED'] = 'false'
    try:
        cfg = GraphitiConfig()
        assert cfg.search.reranker == 'rrf'
        assert cfg.search.mmr_lambda == 0.7
        assert cfg.search.max_facts == 4
        assert cfg.search.exclude_invalidated is False
    finally:
        for k in (
            'SEARCH__RERANKER',
            'SEARCH__MMR_LAMBDA',
            'SEARCH__MAX_FACTS',
            'SEARCH__EXCLUDE_INVALIDATED',
        ):
            os.environ.pop(k, None)


def test_invalid_reranker_rejected():
    os.environ['SEARCH__RERANKER'] = 'bogus'
    try:
        raised = False
        try:
            GraphitiConfig()
        except Exception:
            raised = True
        assert raised, 'expected validation error for unknown reranker'
    finally:
        os.environ.pop('SEARCH__RERANKER', None)
