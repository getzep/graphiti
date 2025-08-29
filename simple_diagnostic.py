#!/usr/bin/env python3
"""
Simple network diagnostic for the intermittent connectivity issue
"""

import subprocess
import time
import socket
from datetime import datetime

def simple_diagnostic():
    hostname = "dinkal-poc-teams.yazge.aktekbilisim.com"
    url = f"https://{hostname}/"
    
    print(f"🔍 Simple Network Diagnostic for {hostname}")
    print(f"⏰ {datetime.now().strftime('%H:%M:%S')}")
    print("-" * 60)
    
    # 1. DNS Resolution Test
    print("1️⃣ DNS Resolution Test:")
    try:
        start_time = time.time()
        result = socket.getaddrinfo(hostname, None)
        dns_time = (time.time() - start_time) * 1000
        ips = list(set([str(ip[4][0]) for ip in result]))
        print(f"   ✅ Resolved to: {', '.join(ips)} ({dns_time:.2f}ms)")
        target_ip = ips[0]
    except Exception as e:
        print(f"   ❌ DNS Failed: {e}")
        return
    
    # 2. Ping Test
    print("2️⃣ Ping Test:")
    try:
        result = subprocess.run(['ping', '-c', '3', hostname], 
                               capture_output=True, text=True, timeout=15)
        if result.returncode == 0:
            lines = result.stdout.split('\n')
            for line in lines:
                if 'packet loss' in line:
                    print(f"   ✅ {line.strip()}")
                    break
        else:
            print(f"   ❌ Ping failed")
    except Exception as e:
        print(f"   ❌ Ping error: {e}")
    
    # 3. TCP Connection Test (Port 443)
    print("3️⃣ TCP Connection Test (443):")
    try:
        start_time = time.time()
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.settimeout(10)
        result = sock.connect_ex((hostname, 443))
        connect_time = (time.time() - start_time) * 1000
        sock.close()
        
        if result == 0:
            print(f"   ✅ TCP connection successful ({connect_time:.2f}ms)")
        else:
            print(f"   ❌ TCP connection failed (error code: {result})")
    except Exception as e:
        print(f"   ❌ TCP test error: {e}")
    
    # 4. HTTP Test with curl
    print("4️⃣ HTTP Test with curl:")
    try:
        result = subprocess.run([
            'curl', '-I', '-s', '-m', '10', url
        ], capture_output=True, text=True, timeout=15)
        
        if result.returncode == 0:
            first_line = result.stdout.split('\n')[0]
            print(f"   ✅ HTTP Response: {first_line}")
            
            # Server header'ını bul
            for line in result.stdout.split('\n'):
                if line.lower().startswith('server:'):
                    print(f"   🖥️ {line}")
                    break
        else:
            print(f"   ❌ HTTP request failed")
            print(f"   Error: {result.stderr}")
    except Exception as e:
        print(f"   ❌ HTTP test error: {e}")
    
    # 5. Traefik Specific Check
    print("5️⃣ Traefik Headers Check:")
    try:
        result = subprocess.run([
            'curl', '-I', '-s', '-m', '10', 
            '-H', 'User-Agent: NetworkDiagnostic/1.0',
            url
        ], capture_output=True, text=True, timeout=15)
        
        if result.returncode == 0:
            traefik_headers = []
            for line in result.stdout.split('\n'):
                line_lower = line.lower()
                if any(header in line_lower for header in ['traefik', 'x-forwarded', 'x-real-ip']):
                    traefik_headers.append(line.strip())
            
            if traefik_headers:
                print(f"   ✅ Traefik headers found:")
                for header in traefik_headers:
                    print(f"      {header}")
            else:
                print(f"   ℹ️ No Traefik-specific headers found")
        else:
            print(f"   ❌ Headers check failed")
    except Exception as e:
        print(f"   ❌ Headers check error: {e}")
    
    # 6. Connection timing test
    print("6️⃣ Connection Timing Test (5 attempts):")
    for i in range(5):
        try:
            start_time = time.time()
            result = subprocess.run([
                'curl', '-I', '-s', '-m', '5', '--connect-timeout', '3', url
            ], capture_output=True, text=True, timeout=8)
            
            timing = (time.time() - start_time) * 1000
            
            if result.returncode == 0:
                status = result.stdout.split('\n')[0].split()[1] if result.stdout else 'Unknown'
                print(f"   ✅ Attempt {i+1}: {status} ({timing:.2f}ms)")
            else:
                print(f"   ❌ Attempt {i+1}: Failed ({timing:.2f}ms)")
                
            time.sleep(1)  # 1 saniye bekle
            
        except Exception as e:
            print(f"   ❌ Attempt {i+1}: Error - {e}")
    
    print("\n" + "=" * 60)
    print("🔍 DIAGNOSTIC SUMMARY:")
    print("   Check the results above for patterns")
    print("   If TCP works but HTTP fails intermittently,")
    print("   the issue might be at the Traefik/application layer")
    print("=" * 60)

if __name__ == "__main__":
    simple_diagnostic()
