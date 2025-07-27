import logging
from datetime import datetime
from typing import TYPE_CHECKING, Any

from helix import Client

from graphiti_core.driver.driver import GraphDriver, GraphDriverSession

logger = logging.getLogger(__name__)


class HelixDriverSession(GraphDriverSession):
    def __init__(self, client: Client):
        self.client = client

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc, tb):
        # No cleanup needed for HelixDB, but method must exist
        pass

    async def close(self):
        # No explicit close needed for HelixDB, but method must exist
        pass

    async def execute_write(self, func, *args, **kwargs):
        # Directly await the provided async function with `self` as the transaction/session
        return await func(self, *args, **kwargs)

    async def _helix_query(self, query: str, params: dict[str, Any]) -> dict[str, Any]:
        result = self.client.query(query, params)  # type: ignore[reportUnknownArgumentType]
        if isinstance(result, list):
            if len(result) == 1:
                result = result[0]
            else:
                response: dict[str, Any] = {str(i):v for i, v in enumerate(result)}
                return response
        return result

    async def run(self, query: str | list, **kwargs: Any) -> Any:
        # HelixDB does not support argument for Label Set, so it's converted into an array of queries
        queries: dict[str, list[dict[str, Any]]] = {}
        if isinstance(query, list):
            for cypher, params in query:
                # Ensure params is a dictionary
                if not isinstance(params, dict):
                    params = {}
                converted_params = convert_datetimes_to_strings(params)
                
                # Type assertion to help the type checker
                if not isinstance(converted_params, dict):
                    continue
                
                # Check if 'query' key exists in params
                if 'query' not in converted_params:
                    raise ValueError("Missing 'query' parameter")
                    
                endpoint = converted_params['query']
                filtered_params = {k: v for k, v in converted_params.items() if k != 'query'}
                if endpoint not in queries:
                    queries[endpoint] = [filtered_params]
                else:
                    queries[endpoint].append(filtered_params)

            for endpoint, params in queries.items():
                await self._helix_query(endpoint, params)  # type: ignore[reportUnknownArgumentType]
        else:
            params = dict(kwargs)
            params = convert_datetimes_to_strings(params)
            
            # Check if 'query' key exists in params
            if 'query' not in params:
                raise ValueError("Missing 'query' parameter")
                
            await self._helix_query(params['query'], {k: v for k, v in params.items() if k != 'query'})  # type: ignore[reportUnknownArgumentType]
        # Assuming `graph.query` is async (ideal); otherwise, wrap in executor
        return None

class HelixDriver(GraphDriver):
    provider: str = 'helixdb'

    def __init__(
        self,
        local: bool = True,
        port: int = 6969,
        api_endpoint: str = "",
        verbose: bool = True,
    ):
        """
        Initialize the HelixDB driver.

        HelixDB is a graph database.
        To connect, provide the host and port.
        The default parameters assume a local (on-premises) HelixDB instance.
        """
        super().__init__()

        self.client = Client(local=local, port=port, api_endpoint=api_endpoint, verbose=verbose)

    async def execute_query(self, cypher_query_, **kwargs: Any):
        # Convert datetime objects to ISO strings (HelixDB does not support datetime objects directly)
        converted_params = convert_datetimes_to_strings(dict(kwargs))

        # Type assertion to help the type checker
        if not isinstance(converted_params, dict):
            raise ValueError("Invalid parameters format")

        # Check if 'query' key exists in params
        if 'query' not in converted_params:
            raise ValueError("Missing 'query' parameter")

        try:
            result = await self._helix_query(converted_params['query'], {k: v for k, v in converted_params.items() if k != 'query'})  # type: ignore[reportUnknownArgumentType]
        except Exception as e:
            logger.error(f'Error executing HelixDB query: {e}')
            raise

        return result

    async def _helix_query(self, query: str, params: dict[str, Any]) -> dict[str, Any]:
        result = self.client.query(query, params)  # type: ignore[reportUnknownArgumentType]
        if isinstance(result, list):
            if len(result) == 1:
                result = result[0]
            else:
                response: dict[str, Any] = {str(i):v for i, v in enumerate(result)}
                return response
        return result

    def session(self, database: str | None = None) -> GraphDriverSession:
        return HelixDriverSession(self.client)

    async def close(self) -> None:
        """Close the driver connection."""
        return None

    async def delete_all_indexes(self) -> None:
        await self._helix_query('deleteAll', {})
        return None

    def clone(self, database: str) -> 'GraphDriver':
        """
        Returns a shallow copy of this driver with a different default database.
        Reuses the same connection (e.g. HelixDB, Neo4j, FalkorDB).
        """
        cloned = HelixDriver(local=self.client.local, port=self.client.h_server_port, api_endpoint=self.client.h_server_api_endpoint, verbose=self.client.verbose)

        return cloned


def convert_datetimes_to_strings(obj):
    if isinstance(obj, dict):
        return {k: convert_datetimes_to_strings(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_datetimes_to_strings(item) for item in obj]
    elif isinstance(obj, tuple):
        return tuple(convert_datetimes_to_strings(item) for item in obj)
    elif isinstance(obj, datetime):
        return obj.isoformat()
    else:
        return obj