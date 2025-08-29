#!/usr/bin/env python3
"""
Quick network test for dinkal-poc-teams.yazge.aktekbilisim.com
"""

import asyncio
import aiohttp
import time
import socket
from datetime import datetime

async def quick_test():
    url = "https://dinkal-poc-teams.yazge.aktekbilisim.com/"
    hostname = "dinkal-poc-teams.yazge.aktekbilisim.com"
    
    print(f"🔍 Quick network test for {url}")
    print(f"⏰ {datetime.now().strftime('%H:%M:%S')}")
    print("-" * 50)
    
    # DNS Test
    try:
        start_time = time.time()
        ips = socket.getaddrinfo(hostname, None)
        dns_time = (time.time() - start_time) * 1000
        ip_list = list(set([ip[4][0] for ip in ips]))
        print(f"✅ DNS Resolution: {dns_time:.2f}ms")
        print(f"   IPs: {', '.join(ip_list)}")
    except Exception as e:
        print(f"❌ DNS Resolution Failed: {e}")
        return
    
    # HTTP Test
    timeout = aiohttp.ClientTimeout(total=10, connect=5)
    try:
        start_time = time.time()
        async with aiohttp.ClientSession(timeout=timeout) as session:
            async with session.head(url) as response:
                response_time = (time.time() - start_time) * 1000
                print(f"✅ HTTP Response: {response.status} ({response_time:.2f}ms)")
                print(f"   Server: {response.headers.get('server', 'Unknown')}")
                print(f"   Date: {response.headers.get('date', 'Unknown')}")
    except asyncio.TimeoutError:
        print("❌ HTTP Request: Timeout")
    except aiohttp.ClientConnectorError as e:
        print(f"❌ HTTP Request: Connection Error - {e}")
    except Exception as e:
        print(f"❌ HTTP Request: Error - {e}")

if __name__ == "__main__":
    asyncio.run(quick_test())
