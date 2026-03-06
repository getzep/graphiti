from __future__ import annotations

import importlib
import sys
import types
from contextlib import suppress
from enum import Enum
from pathlib import Path

_MISSING = object()


def _set_module(name: str, module: types.ModuleType, originals: dict[str, object]) -> None:
    if name not in originals:
        originals[name] = sys.modules.get(name, _MISSING)
    sys.modules[name] = module


def _restore_modules(originals: dict[str, object]) -> None:
    for name, original in originals.items():
        if original is _MISSING:
            sys.modules.pop(name, None)
        else:
            sys.modules[name] = original


def _install_mcp_stubs(originals: dict[str, object]) -> None:
    """Install minimal MCP stubs for unit-test environments without mcp-sdk."""
    try:
        import mcp.server.fastmcp  # noqa: F401
        from mcp.server.auth.middleware.auth_context import get_access_token  # noqa: F401
        return
    except ModuleNotFoundError:
        pass

    mcp_module = types.ModuleType('mcp')
    server_module = types.ModuleType('mcp.server')
    fastmcp_module = types.ModuleType('mcp.server.fastmcp')
    auth_module = types.ModuleType('mcp.server.auth')
    middleware_module = types.ModuleType('mcp.server.auth.middleware')
    auth_context_module = types.ModuleType('mcp.server.auth.middleware.auth_context')

    class Context:  # noqa: D401 - minimal test shim
        """Minimal FastMCP Context stand-in for import-time type references."""

    class FastMCP:  # noqa: D401 - minimal test shim
        """Minimal FastMCP stand-in that preserves decorator behavior."""

        def __init__(self, *_args, **_kwargs):
            self.settings = types.SimpleNamespace(host='127.0.0.1', port=0)

        def tool(self, *_args, **_kwargs):
            def _decorator(func):
                return func

            return _decorator

        def custom_route(self, *_args, **_kwargs):
            def _decorator(func):
                return func

            return _decorator

        async def run_stdio_async(self) -> None:
            return None

        async def run_sse_async(self) -> None:
            return None

        async def run_streamable_http_async(self) -> None:
            return None

    def get_access_token(*_args, **_kwargs):
        return None

    fastmcp_module.Context = Context
    fastmcp_module.FastMCP = FastMCP
    auth_context_module.get_access_token = get_access_token

    mcp_module.server = server_module
    server_module.fastmcp = fastmcp_module
    server_module.auth = auth_module
    auth_module.middleware = middleware_module
    middleware_module.auth_context = auth_context_module

    _set_module('mcp', mcp_module, originals)
    _set_module('mcp.server', server_module, originals)
    _set_module('mcp.server.fastmcp', fastmcp_module, originals)
    _set_module('mcp.server.auth', auth_module, originals)
    _set_module('mcp.server.auth.middleware', middleware_module, originals)
    _set_module('mcp.server.auth.middleware.auth_context', auth_context_module, originals)


def _install_starlette_stub(originals: dict[str, object]) -> None:
    try:
        from starlette.responses import JSONResponse  # noqa: F401
        return
    except ModuleNotFoundError:
        pass

    starlette_module = types.ModuleType('starlette')
    responses_module = types.ModuleType('starlette.responses')

    class JSONResponse(dict):
        def __init__(self, content=None, status_code: int = 200, *_args, **_kwargs):
            super().__init__(content=content, status_code=status_code)

    responses_module.JSONResponse = JSONResponse
    starlette_module.responses = responses_module

    _set_module('starlette', starlette_module, originals)
    _set_module('starlette.responses', responses_module, originals)


def _install_config_schema_stub(originals: dict[str, object]) -> None:
    try:
        import config.schema  # noqa: F401
        return
    except ModuleNotFoundError:
        pass

    config_module = types.ModuleType('config')
    schema_module = types.ModuleType('config.schema')

    class _ConfigBase:
        def __init__(self, **kwargs):
            for key, value in kwargs.items():
                setattr(self, key, value)

    class Neo4jProviderConfig(_ConfigBase):
        pass

    class FalkorDBProviderConfig(_ConfigBase):
        pass

    class LLMConfig(_ConfigBase):
        def __init__(self, provider: str = 'openai', model: str = 'gpt-4o-mini', **kwargs):
            super().__init__(
                provider=provider,
                model=model,
                providers=types.SimpleNamespace(
                    openai=types.SimpleNamespace(api_key='test', api_url=None),
                    azure_openai=None,
                    anthropic=None,
                    gemini=None,
                    groq=None,
                ),
                temperature=0.0,
                max_tokens=1024,
                **kwargs,
            )

    class EmbedderConfig(_ConfigBase):
        def __init__(self, provider: str = 'openai', model: str = 'text-embedding-3-small', **kwargs):
            super().__init__(
                provider=provider,
                model=model,
                providers=types.SimpleNamespace(
                    openai=types.SimpleNamespace(api_key='test', api_url=None),
                    azure_openai=None,
                    gemini=None,
                    voyage=None,
                ),
                **kwargs,
            )

    class DatabaseConfig(_ConfigBase):
        def __init__(self, provider: str = 'neo4j', **kwargs):
            super().__init__(
                provider=provider,
                providers=types.SimpleNamespace(
                    neo4j=Neo4jProviderConfig(uri='bolt://localhost:7687'),
                    falkordb=FalkorDBProviderConfig(host='localhost', port=6379),
                ),
                **kwargs,
            )

    class GraphitiConfig(_ConfigBase):  # noqa: D401 - minimal test shim
        """Minimal config object used by graphiti_mcp_server unit tests."""

        def __init__(self):
            super().__init__(
                database=DatabaseConfig(),
                graphiti=types.SimpleNamespace(
                    group_id='s1_sessions_main',
                    lane_aliases={
                        'sessions_main': ['s1_sessions_main'],
                        'observational_memory': ['s1_observational_memory'],
                        'curated': ['s1_curated_refs'],
                    },
                ),
                server=types.SimpleNamespace(host='127.0.0.1', port=0, transport='stdio'),
                llm=LLMConfig(),
                embedder=EmbedderConfig(),
            )

    class ServerConfig(_ConfigBase):  # noqa: D401 - minimal test shim
        """Minimal server config type for annotation compatibility."""

        def __init__(self):
            super().__init__(host='127.0.0.1', port=0, transport='stdio')

    schema_module.DatabaseConfig = DatabaseConfig
    schema_module.EmbedderConfig = EmbedderConfig
    schema_module.FalkorDBProviderConfig = FalkorDBProviderConfig
    schema_module.GraphitiConfig = GraphitiConfig
    schema_module.LLMConfig = LLMConfig
    schema_module.Neo4jProviderConfig = Neo4jProviderConfig
    schema_module.ServerConfig = ServerConfig
    config_module.schema = schema_module

    _set_module('config', config_module, originals)
    _set_module('config.schema', schema_module, originals)


def _install_graphiti_core_stubs(originals: dict[str, object]) -> None:
    """Install lightweight graphiti_core shims when optional deps are absent."""
    try:
        import graphiti_core as graphiti_core_module
    except ModuleNotFoundError:
        graphiti_core_module = types.ModuleType('graphiti_core')
        graphiti_core_module.__path__ = []
        _set_module('graphiti_core', graphiti_core_module, originals)

    if not hasattr(graphiti_core_module, 'Graphiti'):

        class Graphiti:  # noqa: D401 - minimal test shim
            """Minimal Graphiti stand-in for import-time references."""

            def __init__(self, *_args, **_kwargs):
                self.driver = object()

            async def build_indices_and_constraints(self) -> None:
                return None

            async def add_episode(self, *_args, **_kwargs):
                return None

            async def close(self) -> None:
                return None

            async def search_(self, *_args, **_kwargs):
                return types.SimpleNamespace(nodes=[], edges=[])

        graphiti_core_module.Graphiti = Graphiti

    try:
        import graphiti_core.edges  # noqa: F401
    except Exception:
        edges_module = types.ModuleType('graphiti_core.edges')

        class EntityEdge:  # noqa: D401 - minimal test shim
            """Entity edge test shim."""

            def __init__(self, **kwargs):
                self._payload = kwargs

            def model_dump(self, **_kwargs):
                return dict(self._payload)

            @staticmethod
            async def get_by_uuid(*_args, **_kwargs):
                return None

        edges_module.EntityEdge = EntityEdge
        _set_module('graphiti_core.edges', edges_module, originals)

    try:
        import graphiti_core.nodes  # noqa: F401
    except Exception:
        nodes_module = types.ModuleType('graphiti_core.nodes')

        class EpisodeType(Enum):
            text = 'text'

        class EntityNode:  # noqa: D401 - minimal test shim
            """Entity node test shim."""

            def __init__(self, **kwargs):
                self._payload = kwargs

            def model_dump(self, **_kwargs):
                return dict(self._payload)

        class EpisodicNode:  # noqa: D401 - minimal test shim
            """Episodic node test shim."""

            @staticmethod
            async def get_by_uuid(*_args, **_kwargs):
                return None

            @staticmethod
            async def get_by_group_ids(*_args, **_kwargs):
                return []

        nodes_module.EpisodeType = EpisodeType
        nodes_module.EntityNode = EntityNode
        nodes_module.EpisodicNode = EpisodicNode
        _set_module('graphiti_core.nodes', nodes_module, originals)

    try:
        import graphiti_core.search.search_config  # noqa: F401
        import graphiti_core.search.search_config_recipes  # noqa: F401
    except Exception:
        try:
            import graphiti_core.search as search_module  # noqa: F401
        except Exception:
            search_module = types.ModuleType('graphiti_core.search')
            search_module.__path__ = []
            _set_module('graphiti_core.search', search_module, originals)

        search_config_module = types.ModuleType('graphiti_core.search.search_config')
        search_config_recipes_module = types.ModuleType(
            'graphiti_core.search.search_config_recipes'
        )

        class SearchConfig:  # noqa: D401 - minimal test shim
            """Search config shim with model_copy compatibility."""

            def __init__(self, **kwargs):
                self.__dict__.update(kwargs)

            def model_copy(self, *, update: dict | None = None):
                cloned = SearchConfig(**self.__dict__)
                if update:
                    cloned.__dict__.update(update)
                return cloned

        class NodeSearchConfig:
            def __init__(self, **kwargs):
                self.__dict__.update(kwargs)

        class EdgeSearchConfig:
            def __init__(self, **kwargs):
                self.__dict__.update(kwargs)

        class EdgeSearchMethod(Enum):
            cosine_similarity = 'cosine_similarity'
            bm25 = 'bm25'

        class EdgeReranker(Enum):
            rrf = 'rrf'
            node_distance = 'node_distance'

        search_config_module.SearchConfig = SearchConfig
        search_config_module.NodeSearchConfig = NodeSearchConfig
        search_config_module.EdgeSearchConfig = EdgeSearchConfig
        search_config_module.EdgeSearchMethod = EdgeSearchMethod
        search_config_module.EdgeReranker = EdgeReranker

        search_config_recipes_module.NODE_HYBRID_SEARCH_RRF = SearchConfig(
            node_config=NodeSearchConfig()
        )
        search_config_recipes_module.EDGE_HYBRID_SEARCH_RRF = SearchConfig(
            edge_config=EdgeSearchConfig()
        )
        search_config_recipes_module.EDGE_HYBRID_SEARCH_NODE_DISTANCE = SearchConfig(
            edge_config=EdgeSearchConfig()
        )

        _set_module('graphiti_core.search.search_config', search_config_module, originals)
        _set_module(
            'graphiti_core.search.search_config_recipes',
            search_config_recipes_module,
            originals,
        )

    try:
        import graphiti_core.utils.maintenance.graph_data_operations  # noqa: F401
    except Exception:
        utils_module = types.ModuleType('graphiti_core.utils')
        utils_module.__path__ = []
        maintenance_module = types.ModuleType('graphiti_core.utils.maintenance')
        maintenance_module.__path__ = []
        graph_data_ops_module = types.ModuleType(
            'graphiti_core.utils.maintenance.graph_data_operations'
        )

        async def clear_data(*_args, **_kwargs):
            return None

        graph_data_ops_module.clear_data = clear_data
        _set_module('graphiti_core.utils', utils_module, originals)
        _set_module('graphiti_core.utils.maintenance', maintenance_module, originals)
        _set_module(
            'graphiti_core.utils.maintenance.graph_data_operations',
            graph_data_ops_module,
            originals,
        )


def _install_services_factories_stub(originals: dict[str, object]) -> None:
    """Stub services.factories when graphiti_core provider deps are unavailable."""
    try:
        import services.factories  # noqa: F401
        return
    except Exception:
        pass

    factories_module = types.ModuleType('services.factories')

    class LLMClientFactory:
        @staticmethod
        def create(_config):
            return object()

    class EmbedderFactory:
        @staticmethod
        def create(_config):
            return object()

    class DatabaseDriverFactory:
        @staticmethod
        def create_config(_config):
            return {
                'uri': 'bolt://localhost:7687',
                'user': 'neo4j',
                'password': 'test',
                'host': 'localhost',
                'port': 6379,
                'database': 'default',
            }

    factories_module.LLMClientFactory = LLMClientFactory
    factories_module.EmbedderFactory = EmbedderFactory
    factories_module.DatabaseDriverFactory = DatabaseDriverFactory
    _set_module('services.factories', factories_module, originals)


def load_graphiti_mcp_server():
    module_name = 'mcp_server.src.graphiti_mcp_server'
    already_loaded = sys.modules.get(module_name)
    if already_loaded is not None:
        return already_loaded

    repo_root = Path(__file__).resolve().parents[1]
    mcp_src = str(repo_root / 'mcp_server' / 'src')
    inserted_path = False
    if mcp_src not in sys.path:
        sys.path.insert(0, mcp_src)
        inserted_path = True

    originals: dict[str, object] = {}
    try:
        _install_config_schema_stub(originals)
        _install_mcp_stubs(originals)
        _install_starlette_stub(originals)
        _install_graphiti_core_stubs(originals)
        _install_services_factories_stub(originals)
        return importlib.import_module(module_name)
    finally:
        _restore_modules(originals)
        if inserted_path:
            with suppress(ValueError):
                sys.path.remove(mcp_src)


def load_search_service():
    module_name = 'services.search_service'
    already_loaded = sys.modules.get(module_name)
    if already_loaded is not None:
        return already_loaded

    repo_root = Path(__file__).resolve().parents[1]
    mcp_src = str(repo_root / 'mcp_server' / 'src')
    inserted_path = False
    if mcp_src not in sys.path:
        sys.path.insert(0, mcp_src)
        inserted_path = True

    try:
        return importlib.import_module(module_name)
    finally:
        if inserted_path:
            with suppress(ValueError):
                sys.path.remove(mcp_src)
