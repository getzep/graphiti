#!/usr/bin/env python3
"""
Test script for OAuth endpoints
"""

import httpx
import json

async def test_oauth_endpoints():
    """Test all OAuth endpoints"""
    base_url = "http://localhost:8020"
    
    async with httpx.AsyncClient() as client:
        print("🔍 Testing OAuth endpoints...")
        
        # Test OAuth authorization server metadata
        print("\n1. Testing OAuth authorization server metadata:")
        response = await client.get(f"{base_url}/.well-known/oauth-authorization-server")
        print(f"   Status: {response.status_code}")
        if response.status_code == 200:
            print(f"   ✅ OAuth server metadata: {json.dumps(response.json(), indent=2)}")
        else:
            print(f"   ❌ Failed: {response.text}")
        
        # Test OAuth protected resource metadata  
        print("\n2. Testing OAuth protected resource metadata:")
        response = await client.get(f"{base_url}/.well-known/oauth-protected-resource")
        print(f"   Status: {response.status_code}")
        if response.status_code == 200:
            print(f"   ✅ Protected resource metadata: {json.dumps(response.json(), indent=2)}")
        else:
            print(f"   ❌ Failed: {response.text}")
        
        # Test client registration
        print("\n3. Testing client registration:")
        client_data = {"client_name": "Test OAuth Client", "redirect_uris": ["http://localhost:3000/callback"]}
        response = await client.post(f"{base_url}/register", json=client_data)
        print(f"   Status: {response.status_code}")
        if response.status_code == 201:
            client_info = response.json()
            print(f"   ✅ Client registered: {client_info['client_id']}")
            print(f"   Client secret: {client_info['client_secret'][:10]}...")
        else:
            print(f"   ❌ Failed: {response.text}")
        
        # Test SSE endpoint accessibility (should get timeout, which is expected)
        print("\n4. Testing SSE endpoint accessibility:")
        try:
            response = await client.get(f"{base_url}/sse", timeout=2.0)
            print(f"   Status: {response.status_code}")
        except httpx.ReadTimeout:
            print("   ✅ SSE endpoint accessible (timeout expected for persistent connection)")
        except Exception as e:
            print(f"   ❌ SSE endpoint error: {e}")

if __name__ == "__main__":
    import asyncio
    asyncio.run(test_oauth_endpoints())