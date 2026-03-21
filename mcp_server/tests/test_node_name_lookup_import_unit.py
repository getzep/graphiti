import importlib
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import graphiti_core.helpers as helpers


def test_node_name_lookup_imports_without_validate_group_ids(monkeypatch):
    sys.modules.pop('utils.node_name_lookup', None)
    monkeypatch.delattr(helpers, 'validate_group_ids', raising=False)

    module = importlib.import_module('utils.node_name_lookup')

    assert callable(module.search_nodes_by_name_fallback)
