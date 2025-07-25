#!/usr/bin/env python3
"""Test script to verify OAuth routes are accessible."""

import asyncio
import aiohttp
import sys

async def test_oauth_routes():
    """Test all OAuth routes."""
    base_url = "http://localhost:8020"
    routes = [
        ("GET", "/"),
        ("GET", "/.well-known/oauth-protected-resource"),
        ("GET", "/.well-known/oauth-authorization-server"),
        ("POST", "/register"),
        ("GET", "/sse"),  # Also test the SSE endpoint
    ]

    async with aiohttp.ClientSession() as session:
        for method, path in routes:
            url = f"{base_url}{path}"
            try:
                if method == "GET":
                    async with session.get(url) as response:
                        print(f"{method} {path}: {response.status}")
                        if response.status == 200:
                            content = await response.text()
                            print(f"  Response: {content[:100]}...")
                else:
                    async with session.post(url) as response:
                        print(f"{method} {path}: {response.status}")
                        if response.status == 200:
                            content = await response.text()
                            print(f"  Response: {content[:100]}...")
            except Exception as e:
                print(f"{method} {path}: ERROR - {str(e)}")
            print()

if __name__ == "__main__":
    asyncio.run(test_oauth_routes())
