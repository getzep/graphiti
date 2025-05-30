"""
Automated test script for the Graphiti Browser Agent API.
This script runs a series of tests against the server API endpoints to verify functionality.
"""

import asyncio
import json
import sys
import aiohttp
import argparse
from datetime import datetime
from colorama import Fore, Style, init

# Initialize colorama for colored output
init()

class BrowserAgentAPITester:
    def __init__(self, base_url):
        self.base_url = base_url
        self.test_results = []
        self.session = None
    
    async def setup(self):
        self.session = aiohttp.ClientSession()
    
    async def teardown(self):
        if self.session:
            await self.session.close()
    
    def log_result(self, test_name, passed, message="", details=None):
        result = {
            "test_name": test_name,
            "passed": passed,
            "message": message,
            "details": details,
            "timestamp": datetime.now().isoformat()
        }
        self.test_results.append(result)
        
        status = f"{Fore.GREEN}PASS{Style.RESET_ALL}" if passed else f"{Fore.RED}FAIL{Style.RESET_ALL}"
        print(f"[{status}] {test_name}")
        if message:
            print(f"  {message}")
        if details:
            print(f"  Details: {json.dumps(details, indent=2)}")
    
    async def test_health_endpoint(self):
        """Test the health check endpoint."""
        test_name = "Health Check Endpoint"
        url = f"{self.base_url}/browser_agent/health"
        
        try:
            async with self.session.get(url) as response:
                if response.status == 200:
                    data = await response.json()
                    if data.get("status") == "ok" and data.get("service") == "browser_agent":
                        self.log_result(test_name, True, "Health check endpoint returned correct response", data)
                    else:
                        self.log_result(test_name, False, "Health check endpoint returned unexpected data", data)
                else:
                    error_text = await response.text()
                    self.log_result(test_name, False, f"Health check endpoint returned status {response.status}", error_text)
        except Exception as e:
            self.log_result(test_name, False, f"Exception during health check: {str(e)}")

    async def test_categorize_endpoint(self):
        """Test the categorize endpoint with sample data."""
        test_name = "Categorize Endpoint"
        url = f"{self.base_url}/browser_agent/categorize"
        
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
        
        try:
            async with self.session.post(url, json=sample_data) as response:
                if response.status == 200:
                    data = await response.json()
                    if "categories" in data and isinstance(data["categories"], list) and len(data["categories"]) > 0:
                        self.log_result(test_name, True, f"Categorize endpoint returned {len(data['categories'])} categories", data)
                    else:
                        self.log_result(test_name, False, "Categorize endpoint returned invalid categories", data)
                else:
                    error_text = await response.text()
                    self.log_result(test_name, False, f"Categorize endpoint returned status {response.status}", error_text)
        except Exception as e:
            self.log_result(test_name, False, f"Exception during categorization: {str(e)}")

    async def test_ingest_endpoint(self):
        """Test the ingest endpoint with sample data."""
        test_name = "Ingest Endpoint"
        url = f"{self.base_url}/browser_agent/ingest"
        
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
        
        try:
            async with self.session.post(url, json=sample_data) as response:
                if response.status == 200:
                    data = await response.json()
                    if data.get("status") == "success" and "episode_id" in data:
                        self.log_result(test_name, True, "Ingest endpoint successfully added data to knowledge graph", data)
                    else:
                        self.log_result(test_name, False, "Ingest endpoint returned unexpected data", data)
                else:
                    error_text = await response.text()
                    self.log_result(test_name, False, f"Ingest endpoint returned status {response.status}", error_text)
        except Exception as e:
            self.log_result(test_name, False, f"Exception during ingestion: {str(e)}")

    async def test_cors_headers(self):
        """Test that CORS headers are properly set."""
        test_name = "CORS Headers"
        url = f"{self.base_url}/browser_agent/health"
        
        try:
            async with self.session.options(url) as response:
                if response.status == 200:
                    # Check for CORS headers
                    headers = response.headers
                    if "Access-Control-Allow-Origin" in headers:
                        self.log_result(test_name, True, "CORS headers are properly set", dict(headers))
                    else:
                        self.log_result(test_name, False, "CORS headers are missing", dict(headers))
                else:
                    self.log_result(test_name, False, f"OPTIONS request returned status {response.status}")
        except Exception as e:
            self.log_result(test_name, False, f"Exception during CORS test: {str(e)}")

    def print_summary(self):
        """Print a summary of all test results."""
        total_tests = len(self.test_results)
        passed_tests = sum(1 for result in self.test_results if result["passed"])
        failed_tests = total_tests - passed_tests
        
        print("\n" + "=" * 50)
        print(f"TEST SUMMARY: {passed_tests}/{total_tests} tests passed")
        print("=" * 50)
        
        if failed_tests > 0:
            print(f"\n{Fore.RED}Failed Tests:{Style.RESET_ALL}")
            for result in self.test_results:
                if not result["passed"]:
                    print(f"- {result['test_name']}: {result['message']}")
        
        print("\nDetailed Results:")
        for result in self.test_results:
            status = f"{Fore.GREEN}PASS{Style.RESET_ALL}" if result["passed"] else f"{Fore.RED}FAIL{Style.RESET_ALL}"
            print(f"[{status}] {result['test_name']}")
        
        return passed_tests == total_tests

async def main():
    parser = argparse.ArgumentParser(description="Test the Graphiti Browser Agent API")
    parser.add_argument("--url", default="http://localhost:8000", help="Base URL of the Graphiti API server")
    args = parser.parse_args()
    
    print(f"{Fore.CYAN}Testing Graphiti Browser Agent API at {args.url}{Style.RESET_ALL}\n")
    
    tester = BrowserAgentAPITester(args.url)
    await tester.setup()
    
    try:
        # Run all tests
        await tester.test_health_endpoint()
        await tester.test_categorize_endpoint()
        await tester.test_ingest_endpoint()
        await tester.test_cors_headers()
    finally:
        await tester.teardown()
    
    # Print summary and exit with appropriate code
    all_passed = tester.print_summary()
    sys.exit(0 if all_passed else 1)

if __name__ == "__main__":
    asyncio.run(main())