from __future__ import annotations

import json
from itertools import count
from typing import Any

import httpx


def _parse_sse_data(response_text: str) -> dict[str, Any]:
    for line in response_text.splitlines():
        if line.startswith('data: '):
            return json.loads(line[6:])
    raise ValueError(f'No SSE data line found in response: {response_text[:300]}')


class RawHttpMCPClient:
    def __init__(self, base_url: str):
        self.base_url = base_url.rstrip('/')
        self.mcp_url = f'{self.base_url}/mcp'
        self.session_id: str | None = None
        self.client = httpx.AsyncClient(timeout=90.0)
        self._request_ids = count(1)

    async def __aenter__(self) -> RawHttpMCPClient:
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb) -> None:
        await self.client.aclose()

    async def initialize(self) -> dict[str, Any]:
        return await self._post(
            {
                'jsonrpc': '2.0',
                'id': next(self._request_ids),
                'method': 'initialize',
                'params': {
                    'protocolVersion': '2025-03-26',
                    'capabilities': {},
                    'clientInfo': {'name': 'raw-http-test-client', 'version': '1.0'},
                },
            }
        )

    async def list_tools(self) -> dict[str, Any]:
        return await self._post(
            {
                'jsonrpc': '2.0',
                'id': next(self._request_ids),
                'method': 'tools/list',
                'params': {},
            }
        )

    async def call_tool(self, tool_name: str, arguments: dict[str, Any]) -> dict[str, Any]:
        return await self._post(
            {
                'jsonrpc': '2.0',
                'id': next(self._request_ids),
                'method': 'tools/call',
                'params': {'name': tool_name, 'arguments': arguments},
            }
        )

    async def _post(self, payload: dict[str, Any]) -> dict[str, Any]:
        headers = {
            'Content-Type': 'application/json',
            'Accept': 'application/json, text/event-stream',
        }
        if self.session_id:
            headers['mcp-session-id'] = self.session_id

        response = await self.client.post(self.mcp_url, json=payload, headers=headers)
        response.raise_for_status()

        next_session_id = response.headers.get('mcp-session-id')
        if next_session_id:
            self.session_id = next_session_id

        return _parse_sse_data(response.text)
