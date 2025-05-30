"""
Test script for the Graphiti Browser Agent API.
This script simulates requests from the browser extension to test the API endpoints.
"""

import asyncio
import json
import aiohttp
import argparse

async def test_health_endpoint(base_url):
    """Test the health check endpoint."""
    url = f"{base_url}/browser_agent/health"
    
    async with aiohttp.ClientSession() as session:
        async with session.get(url) as response:
            print(f"Health check status: {response.status}")
            if response.status == 200:
                data = await response.json()
                print(f"Response: {json.dumps(data, indent=2)}")
            else:
                print(f"Error: {await response.text()}")

async def test_categorize_endpoint(base_url):
    """Test the categorize endpoint with sample data."""
    url = f"{base_url}/browser_agent/categorize"
    
    # Sample web page content
    sample_data = {
        "content": {
            "url": "https://example.com/article",
            "title": "Understanding Climate Change: A Comprehensive Guide",
            "content": {
                "text": "Climate change is one of the most pressing issues of our time. It refers to long-term shifts in temperatures and weather patterns. These shifts may be natural, but since the 1800s, human activities have been the main driver of climate change, primarily due to the burning of fossil fuels like coal, oil, and gas, which produces heat-trapping gases. This article explores the causes, effects, and potential solutions to climate change.",
                "html": "<article><h1>Understanding Climate Change</h1><p>Climate change is one of the most pressing issues of our time...</p></article>"
            },
            "metaTags": {
                "description": "Learn about the causes and effects of climate change and what we can do to mitigate its impact.",
                "keywords": "climate change, global warming, environment, sustainability"
            }
        }
    }
    
    async with aiohttp.ClientSession() as session:
        async with session.post(url, json=sample_data) as response:
            print(f"Categorize status: {response.status}")
            if response.status == 200:
                data = await response.json()
                print(f"AI Categories: {json.dumps(data, indent=2)}")
            else:
                print(f"Error: {await response.text()}")

async def test_ingest_endpoint(base_url):
    """Test the ingest endpoint with sample data."""
    url = f"{base_url}/browser_agent/ingest"
    
    # Sample web page content with categories
    sample_data = {
        "content": {
            "url": "https://example.com/article",
            "title": "Understanding Climate Change: A Comprehensive Guide",
            "content": {
                "text": "Climate change is one of the most pressing issues of our time. It refers to long-term shifts in temperatures and weather patterns. These shifts may be natural, but since the 1800s, human activities have been the main driver of climate change, primarily due to the burning of fossil fuels like coal, oil, and gas, which produces heat-trapping gases. This article explores the causes, effects, and potential solutions to climate change.",
                "html": "<article><h1>Understanding Climate Change</h1><p>Climate change is one of the most pressing issues of our time...</p></article>"
            },
            "metaTags": {
                "description": "Learn about the causes and effects of climate change and what we can do to mitigate its impact.",
                "keywords": "climate change, global warming, environment, sustainability"
            }
        },
        "categories": [
            "Climate Change",
            "Environment",
            "Sustainability",
            "Global Warming",
            "Science"
        ],
        "source": {
            "url": "https://example.com/article",
            "title": "Understanding Climate Change: A Comprehensive Guide",
            "timestamp": "2025-05-21T12:34:56Z"
        }
    }
    
    async with aiohttp.ClientSession() as session:
        async with session.post(url, json=sample_data) as response:
            print(f"Ingest status: {response.status}")
            if response.status == 200:
                data = await response.json()
                print(f"Response: {json.dumps(data, indent=2)}")
            else:
                print(f"Error: {await response.text()}")

async def main():
    parser = argparse.ArgumentParser(description="Test the Graphiti Browser Agent API")
    parser.add_argument("--url", default="http://localhost:8000", help="Base URL of the Graphiti API server")
    args = parser.parse_args()
    
    print(f"Testing API at {args.url}")
    
    print("\n=== Testing Health Endpoint ===")
    await test_health_endpoint(args.url)
    
    print("\n=== Testing Categorize Endpoint ===")
    await test_categorize_endpoint(args.url)
    
    print("\n=== Testing Ingest Endpoint ===")
    await test_ingest_endpoint(args.url)

if __name__ == "__main__":
    asyncio.run(main())