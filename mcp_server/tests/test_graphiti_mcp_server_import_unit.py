import builtins
import importlib
import sys


def test_graphiti_mcp_server_imports_without_repo_local_graphiti_core_helper(monkeypatch):
    sys.modules.pop('graphiti_mcp_server', None)
    original_import = builtins.__import__

    def _guarded_import(name, globals=None, locals=None, fromlist=(), level=0):
        if name == 'graphiti_core.search.node_name_lookup':
            raise ModuleNotFoundError(name)
        return original_import(name, globals, locals, fromlist, level)

    monkeypatch.setattr(builtins, '__import__', _guarded_import)

    module = importlib.import_module('graphiti_mcp_server')

    assert callable(module.search_nodes_by_name_fallback)
