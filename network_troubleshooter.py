#!/usr/bin/env python3
"""
Advanced troubleshooting script for dinkal-poc-teams.yazge.aktekbilisim.com
Analyzes Traefik routing, DNS issues, and network connectivity problems.
"""

import asyncio
import aiohttp
import subprocess
import time
import json
import socket
from datetime import datetime
from typing import Dict, Any, List
import logging

logger = logging.getLogger(__name__)

class NetworkTroubleshooter:
    def __init__(self, url: str):
        self.url = url
        self.hostname = url.replace('https://', '').replace('http://', '').split('/')[0]
        
    def get_dns_details(self) -> Dict[str, Any]:
        """Detaylı DNS analizi"""
        results = {}
        
        # A record
        try:
            result = subprocess.run(['dig', '+short', 'A', self.hostname], 
                                  capture_output=True, text=True, timeout=10)
            results['A_records'] = result.stdout.strip().split('\n') if result.stdout.strip() else []
        except:
            results['A_records'] = []
            
        # AAAA record
        try:
            result = subprocess.run(['dig', '+short', 'AAAA', self.hostname], 
                                  capture_output=True, text=True, timeout=10)
            results['AAAA_records'] = result.stdout.strip().split('\n') if result.stdout.strip() else []
        except:
            results['AAAA_records'] = []
            
        # CNAME record
        try:
            result = subprocess.run(['dig', '+short', 'CNAME', self.hostname], 
                                  capture_output=True, text=True, timeout=10)
            results['CNAME_records'] = result.stdout.strip().split('\n') if result.stdout.strip() else []
        except:
            results['CNAME_records'] = []
            
        # NS record
        try:
            result = subprocess.run(['dig', '+short', 'NS', self.hostname], 
                                  capture_output=True, text=True, timeout=10)
            results['NS_records'] = result.stdout.strip().split('\n') if result.stdout.strip() else []
        except:
            results['NS_records'] = []
            
        return results
    
    def check_tcp_connectivity(self, port: int = 443) -> Dict[str, Any]:
        """TCP port connectivity kontrolü"""
        try:
            start_time = time.time()
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.settimeout(10)
            result = sock.connect_ex((self.hostname, port))
            connect_time = (time.time() - start_time) * 1000
            sock.close()
            
            return {
                'success': result == 0,
                'port': port,
                'connect_time_ms': round(connect_time, 2),
                'error_code': result if result != 0 else None
            }
        except Exception as e:
            return {
                'success': False,
                'port': port,
                'error': str(e)
            }
    
    def check_ssl_certificate(self) -> Dict[str, Any]:
        """SSL sertifika kontrolü"""
        try:
            result = subprocess.run([
                'openssl', 's_client', '-connect', f'{self.hostname}:443',
                '-servername', self.hostname, '-verify_return_error'
            ], input='\n', capture_output=True, text=True, timeout=15)
            
            # Certificate details parsing
            cert_info = {}
            lines = result.stderr.split('\n')
            
            for line in lines:
                if 'verify return:' in line:
                    cert_info['verify_result'] = line.split('verify return:')[1].strip()
                elif 'subject=' in line:
                    cert_info['subject'] = line.split('subject=')[1].strip()
                elif 'issuer=' in line:
                    cert_info['issuer'] = line.split('issuer=')[1].strip()
                    
            return {
                'success': result.returncode == 0,
                'certificate_info': cert_info,
                'output': result.stderr[:500]  # İlk 500 karakter
            }
        except Exception as e:
            return {
                'success': False,
                'error': str(e)
            }
    
    async def check_http_headers(self) -> Dict[str, Any]:
        """HTTP başlık analizi"""
        timeout = aiohttp.ClientTimeout(total=15)
        
        try:
            async with aiohttp.ClientSession(timeout=timeout) as session:
                async with session.get(self.url, allow_redirects=False) as response:
                    headers = dict(response.headers)
                    
                    # Traefik ile ilgili başlıkları kontrol et
                    traefik_headers = {
                        k: v for k, v in headers.items() 
                        if 'traefik' in k.lower() or 'x-forwarded' in k.lower()
                    }
                    
                    return {
                        'success': True,
                        'status_code': response.status,
                        'all_headers': headers,
                        'traefik_headers': traefik_headers,
                        'server': headers.get('server', 'Unknown'),
                        'content_type': headers.get('content-type', 'Unknown')
                    }
        except Exception as e:
            return {
                'success': False,
                'error': str(e)
            }
    
    def check_network_route(self) -> Dict[str, Any]:
        """Network route analizi"""
        try:
            # Traceroute
            traceroute_result = subprocess.run([
                'traceroute', '-m', '15', self.hostname
            ], capture_output=True, text=True, timeout=60)
            
            # Route parsing
            hops = []
            lines = traceroute_result.stdout.split('\n')[1:]  # İlk satırı atla
            
            for line in lines:
                if line.strip() and not line.startswith(' '):
                    parts = line.split()
                    if len(parts) >= 2:
                        hop_num = parts[0]
                        hop_info = ' '.join(parts[1:])
                        hops.append(f"{hop_num} {hop_info}")
                        
            return {
                'success': True,
                'hops': hops[:15],  # İlk 15 hop
                'full_output': traceroute_result.stdout
            }
        except Exception as e:
            return {
                'success': False,
                'error': str(e)
            }
    
    def check_dns_propagation(self) -> Dict[str, Any]:
        """DNS propagation kontrolü"""
        dns_servers = [
            ('Google', '8.8.8.8'),
            ('Cloudflare', '1.1.1.1'),
            ('OpenDNS', '208.67.222.222'),
            ('Local', None)
        ]
        
        results = {}
        
        for name, server in dns_servers:
            try:
                if server:
                    cmd = ['dig', f'@{server}', '+short', 'A', self.hostname]
                else:
                    cmd = ['dig', '+short', 'A', self.hostname]
                    
                result = subprocess.run(cmd, capture_output=True, text=True, timeout=10)
                ips = result.stdout.strip().split('\n') if result.stdout.strip() else []
                
                results[name] = {
                    'success': len(ips) > 0,
                    'ips': ips,
                    'server': server
                }
            except Exception as e:
                results[name] = {
                    'success': False,
                    'error': str(e),
                    'server': server
                }
                
        return results
    
    async def run_comprehensive_analysis(self) -> Dict[str, Any]:
        """Kapsamlı ağ analizi"""
        print(f"🔬 Comprehensive Network Analysis for {self.url}")
        print(f"⏰ {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print("=" * 70)
        
        analysis: Dict[str, Any] = {
            'timestamp': datetime.now().isoformat(),
            'url': self.url,
            'hostname': self.hostname
        }
        
        # DNS Details
        print("🌐 DNS Analysis...")
        analysis['dns_details'] = self.get_dns_details()
        
        # DNS Propagation
        print("📡 DNS Propagation Check...")
        analysis['dns_propagation'] = self.check_dns_propagation()
        
        # TCP Connectivity
        print("🔌 TCP Connectivity (443)...")
        analysis['tcp_443'] = self.check_tcp_connectivity(443)
        
        print("🔌 TCP Connectivity (80)...")
        analysis['tcp_80'] = self.check_tcp_connectivity(80)
        
        # SSL Certificate
        print("🔒 SSL Certificate Check...")
        analysis['ssl_certificate'] = self.check_ssl_certificate()
        
        # HTTP Headers
        print("📋 HTTP Headers Analysis...")
        analysis['http_headers'] = await self.check_http_headers()
        
        # Network Route
        print("🛣️ Network Route Analysis...")
        analysis['network_route'] = self.check_network_route()
        
        return analysis
    
    def print_analysis_summary(self, analysis: Dict[str, Any]):
        """Analiz özetini yazdır"""
        print("\n" + "=" * 70)
        print("📊 ANALYSIS SUMMARY")
        print("=" * 70)
        
        # DNS Summary
        dns_prop = analysis['dns_propagation']
        print(f"🌐 DNS Status:")
        for provider, data in dns_prop.items():
            status = "✅" if data['success'] else "❌"
            ips = ', '.join(data.get('ips', [])) if data['success'] else data.get('error', 'Failed')
            print(f"   {status} {provider}: {ips}")
        
        # Connectivity Summary
        tcp_443 = analysis['tcp_443']
        tcp_80 = analysis['tcp_80']
        print(f"\n🔌 Connectivity:")
        print(f"   {'✅' if tcp_443['success'] else '❌'} HTTPS (443): {tcp_443.get('connect_time_ms', 'Failed')}ms")
        print(f"   {'✅' if tcp_80['success'] else '❌'} HTTP (80): {tcp_80.get('connect_time_ms', 'Failed')}ms")
        
        # SSL Summary
        ssl = analysis['ssl_certificate']
        print(f"\n🔒 SSL Certificate:")
        print(f"   {'✅' if ssl['success'] else '❌'} Certificate: {'Valid' if ssl['success'] else ssl.get('error', 'Invalid')}")
        
        # HTTP Summary
        http = analysis['http_headers']
        if http['success']:
            print(f"\n📋 HTTP Response:")
            print(f"   ✅ Status: {http['status_code']}")
            print(f"   ✅ Server: {http['server']}")
            if http['traefik_headers']:
                print(f"   🔄 Traefik Headers Found: {len(http['traefik_headers'])}")
                for k, v in http['traefik_headers'].items():
                    print(f"      {k}: {v}")
        else:
            print(f"\n📋 HTTP Response:")
            print(f"   ❌ Failed: {http.get('error', 'Unknown error')}")
        
        # Route Summary
        route = analysis['network_route']
        if route['success'] and route['hops']:
            print(f"\n🛣️ Network Route:")
            print(f"   ✅ Traceroute completed with {len(route['hops'])} hops")
            print(f"   📍 Last few hops:")
            for hop in route['hops'][-3:]:  # Son 3 hop
                print(f"      {hop}")
        else:
            print(f"\n🛣️ Network Route:")
            print(f"   ❌ Traceroute failed: {route.get('error', 'Unknown error')}")

async def main():
    """Ana fonksiyon"""
    url = "https://dinkal-poc-teams.yazge.aktekbilisim.com/"
    troubleshooter = NetworkTroubleshooter(url)
    
    analysis = await troubleshooter.run_comprehensive_analysis()
    troubleshooter.print_analysis_summary(analysis)
    
    # JSON formatında da kaydet
    with open('network_analysis.json', 'w') as f:
        json.dump(analysis, f, indent=2, default=str)
    
    print(f"\n💾 Detailed analysis saved to network_analysis.json")

if __name__ == "__main__":
    asyncio.run(main())
