#!/usr/bin/env python3
"""
Domain vs Traefik problem isolation script
Tests direct IP access vs domain-based access to isolate the issue
"""

import subprocess
import time
import socket
import json
from datetime import datetime
from collections import defaultdict

class DomainTraefikIsolator:
    def __init__(self):
        self.domain = "dinkal-poc-teams.yazge.aktekbilisim.com"
        self.https_url = f"https://{self.domain}/"
        self.http_url = f"http://{self.domain}/"
        self.target_ip = None
        self.results = {
            'domain_https': [],
            'domain_http': [],
            'direct_ip_https': [],
            'direct_ip_http': [],
            'traefik_bypass': []
        }
    
    def get_target_ip(self):
        """Target IP adresini al"""
        try:
            result = socket.getaddrinfo(self.domain, None)
            self.target_ip = result[0][4][0]
            print(f"🎯 Target IP: {self.target_ip}")
            return True
        except Exception as e:
            print(f"❌ DNS resolution failed: {e}")
            return False
    
    def test_request(self, url, test_type, host_header=None):
        """Tek HTTP request testi"""
        try:
            start_time = time.time()
            
            cmd = ['curl', '-I', '-s', '-m', '5', '--connect-timeout', '3']
            
            if host_header:
                cmd.extend(['-H', f'Host: {host_header}'])
            
            cmd.append(url)
            
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=8)
            timing = (time.time() - start_time) * 1000
            
            success = result.returncode == 0
            status_code = None
            server = None
            
            if success and result.stdout:
                lines = result.stdout.split('\n')
                if lines[0]:
                    parts = lines[0].split()
                    if len(parts) >= 2:
                        status_code = parts[1]
                
                for line in lines:
                    if line.lower().startswith('server:'):
                        server = line.split(':', 1)[1].strip()
                        break
            
            return {
                'success': success,
                'status_code': status_code,
                'server': server,
                'response_time_ms': round(timing, 2),
                'timestamp': datetime.now().isoformat(),
                'error': result.stderr if not success else None
            }
            
        except Exception as e:
            return {
                'success': False,
                'error': str(e),
                'response_time_ms': None,
                'timestamp': datetime.now().isoformat()
            }
    
    def test_domain_access(self, iterations=10):
        """Domain üzerinden erişim testi"""
        print(f"\n🌐 DOMAIN ACCESS TEST ({iterations} iterations)")
        print("-" * 50)
        
        for i in range(iterations):
            # HTTPS
            result_https = self.test_request(self.https_url, 'domain_https')
            self.results['domain_https'].append(result_https)
            
            # HTTP  
            result_http = self.test_request(self.http_url, 'domain_http')
            self.results['domain_http'].append(result_http)
            
            status_https = "✅" if result_https['success'] else "❌"
            status_http = "✅" if result_http['success'] else "❌"
            
            print(f"   {i+1:2d}. HTTPS: {status_https} | HTTP: {status_http}")
            
            time.sleep(1)
    
    def test_direct_ip_access(self, iterations=10):
        """Direct IP erişim testi"""
        if not self.target_ip:
            print("❌ No target IP available")
            return
            
        print(f"\n🎯 DIRECT IP ACCESS TEST ({iterations} iterations)")
        print(f"   Target: {self.target_ip}")
        print("-" * 50)
        
        for i in range(iterations):
            # HTTPS with Host header
            https_url = f"https://{self.target_ip}/"
            result_https = self.test_request(https_url, 'direct_ip_https', host_header=self.domain)
            self.results['direct_ip_https'].append(result_https)
            
            # HTTP with Host header
            http_url = f"http://{self.target_ip}/"
            result_http = self.test_request(http_url, 'direct_ip_http', host_header=self.domain)
            self.results['direct_ip_http'].append(result_http)
            
            status_https = "✅" if result_https['success'] else "❌"
            status_http = "✅" if result_http['success'] else "❌"
            
            print(f"   {i+1:2d}. HTTPS: {status_https} | HTTP: {status_http}")
            
            time.sleep(1)
    
    def test_traefik_bypass(self, iterations=10):
        """Traefik bypass testleri"""
        if not self.target_ip:
            return
            
        print(f"\n🔀 TRAEFIK BYPASS TEST ({iterations} iterations)")
        print("   Testing common backend ports...")
        print("-" * 50)
        
        # Common backend ports
        backend_ports = [8080, 3000, 5000, 8000, 9000]
        
        for port in backend_ports:
            print(f"\n   Testing port {port}:")
            success_count = 0
            
            for i in range(3):  # 3 test per port
                try:
                    url = f"http://{self.target_ip}:{port}/"
                    result = self.test_request(url, 'traefik_bypass')
                    self.results['traefik_bypass'].append({
                        **result,
                        'port': port,
                        'attempt': i+1
                    })
                    
                    if result['success']:
                        success_count += 1
                        print(f"      ✅ Port {port} - Attempt {i+1}")
                    else:
                        print(f"      ❌ Port {port} - Attempt {i+1}")
                        
                    time.sleep(0.5)
                    
                except Exception as e:
                    print(f"      ❌ Port {port} - Error: {e}")
            
            if success_count > 0:
                print(f"   🎯 Port {port}: {success_count}/3 successful - POTENTIAL BACKEND!")
    
    def analyze_results(self):
        """Sonuçları analiz et"""
        print(f"\n📊 ANALYSIS RESULTS")
        print("=" * 70)
        
        analysis = {}
        
        for test_type, results in self.results.items():
            if not results:
                continue
                
            total = len(results)
            successful = sum(1 for r in results if r['success'])
            success_rate = (successful / total) * 100 if total > 0 else 0
            
            analysis[test_type] = {
                'total': total,
                'successful': successful,
                'success_rate': success_rate
            }
            
            print(f"{test_type.replace('_', ' ').title():<20}: {successful}/{total} ({success_rate:5.1f}%)")
        
        # Problem isolation logic
        print(f"\n🔍 PROBLEM ISOLATION:")
        print("-" * 50)
        
        domain_https_rate = analysis.get('domain_https', {}).get('success_rate', 0)
        domain_http_rate = analysis.get('domain_http', {}).get('success_rate', 0)
        direct_ip_https_rate = analysis.get('direct_ip_https', {}).get('success_rate', 0)
        direct_ip_http_rate = analysis.get('direct_ip_http', {}).get('success_rate', 0)
        
        # Decision logic
        if domain_https_rate < 80 and direct_ip_https_rate >= 80:
            print("🌐 DOMAIN/DNS PROBLEM:")
            print("   → Domain resolution intermittent")
            print("   → Check DNS configuration")
            print("   → Possible CDN/proxy issues")
            
        elif domain_https_rate < 80 and direct_ip_https_rate < 80:
            print("🔄 TRAEFIK/LOAD BALANCER PROBLEM:")
            print("   → Backend instance issues")
            print("   → Load balancer configuration")
            print("   → Health check problems")
            
        elif domain_https_rate >= 80 and domain_http_rate < 80:
            print("🔒 SSL/HTTPS PROBLEM:")
            print("   → Certificate issues")
            print("   → HTTPS redirect problems")
            
        elif abs(domain_https_rate - domain_http_rate) < 10:
            print("⚖️ CONSISTENT PROBLEM:")
            print("   → Backend application issues")
            print("   → Resource constraints")
            print("   → Database connectivity")
            
        else:
            print("❓ MIXED RESULTS:")
            print("   → Need more investigation")
            print("   → Check application logs")
        
        # Traefik bypass analysis
        bypass_results = self.results.get('traefik_bypass', [])
        if bypass_results:
            successful_ports = defaultdict(int)
            for result in bypass_results:
                if result['success']:
                    successful_ports[result['port']] += 1
            
            if successful_ports:
                print(f"\n🎯 DIRECT BACKEND ACCESS:")
                for port, count in successful_ports.items():
                    print(f"   → Port {port}: {count} successful connections")
                    print(f"     💡 Try: http://{self.target_ip}:{port}/")
        
        return analysis
    
    def save_results(self, filename='domain_traefik_analysis.json'):
        """Sonuçları kaydet"""
        data = {
            'domain': self.domain,
            'target_ip': self.target_ip,
            'test_timestamp': datetime.now().isoformat(),
            'results': self.results,
            'analysis': self.analyze_results()
        }
        
        with open(filename, 'w') as f:
            json.dump(data, f, indent=2, default=str)
        
        print(f"\n💾 Results saved to {filename}")
    
    def run_full_test(self):
        """Tam test süiti"""
        print("🔬 DOMAIN vs TRAEFIK ISOLATION TEST")
        print("=" * 70)
        print(f"⏰ {datetime.now().strftime('%H:%M:%S')}")
        
        # 1. Get target IP
        if not self.get_target_ip():
            return
        
        # 2. Domain access tests
        self.test_domain_access(10)
        
        # 3. Direct IP access tests
        self.test_direct_ip_access(10)
        
        # 4. Traefik bypass tests
        self.test_traefik_bypass(5)
        
        # 5. Analysis
        self.analyze_results()
        
        # 6. Save results
        self.save_results()

def main():
    isolator = DomainTraefikIsolator()
    isolator.run_full_test()

if __name__ == "__main__":
    main()
