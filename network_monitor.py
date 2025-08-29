#!/usr/bin/env python3
"""
Network connectivity monitor for dinkal-poc-teams.yazge.aktekbilisim.com
Monitors the URL recursively and logs connection issues for debugging.
"""

import asyncio
import aiohttp
import time
import socket
import subprocess
import json
from datetime import datetime
from typing import Dict, Any, Optional
import logging
from pathlib import Path

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('network_monitor.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class NetworkMonitor:
    def __init__(self, url: str, check_interval: int = 30):
        self.url = url
        self.check_interval = check_interval
        self.stats = {
            'total_checks': 0,
            'successful_requests': 0,
            'failed_requests': 0,
            'dns_failures': 0,
            'connection_timeouts': 0,
            'http_errors': 0,
            'start_time': datetime.now().isoformat()
        }
        
    async def check_dns_resolution(self, hostname: str) -> Dict[str, Any]:
        """DNS çözümleme kontrolü"""
        try:
            start_time = time.time()
            result = socket.getaddrinfo(hostname, None)
            dns_time = (time.time() - start_time) * 1000
            
            ips = [ip[4][0] for ip in result]
            return {
                'success': True,
                'ips': list(set(ips)),
                'dns_time_ms': round(dns_time, 2)
            }
        except Exception as e:
            return {
                'success': False,
                'error': str(e),
                'dns_time_ms': None
            }
    
    async def check_http_connectivity(self) -> Dict[str, Any]:
        """HTTP bağlantı kontrolü"""
        timeout = aiohttp.ClientTimeout(total=10, connect=5)
        
        try:
            start_time = time.time()
            async with aiohttp.ClientSession(timeout=timeout) as session:
                async with session.head(self.url) as response:
                    response_time = (time.time() - start_time) * 1000
                    
                    return {
                        'success': True,
                        'status_code': response.status,
                        'response_time_ms': round(response_time, 2),
                        'headers': dict(response.headers),
                        'server': response.headers.get('server', 'Unknown')
                    }
                    
        except asyncio.TimeoutError:
            return {
                'success': False,
                'error': 'Connection timeout',
                'error_type': 'timeout'
            }
        except aiohttp.ClientConnectorError as e:
            return {
                'success': False,
                'error': f'Connection error: {str(e)}',
                'error_type': 'connection'
            }
        except Exception as e:
            return {
                'success': False,
                'error': f'HTTP error: {str(e)}',
                'error_type': 'http'
            }
    
    def check_traceroute(self, hostname: str) -> Dict[str, Any]:
        """Traceroute kontrolü (sadece hata durumunda)"""
        try:
            result = subprocess.run(
                ['traceroute', '-m', '10', hostname],
                capture_output=True,
                text=True,
                timeout=30
            )
            return {
                'success': True,
                'output': result.stdout,
                'return_code': result.returncode
            }
        except subprocess.TimeoutExpired:
            return {
                'success': False,
                'error': 'Traceroute timeout'
            }
        except Exception as e:
            return {
                'success': False,
                'error': f'Traceroute error: {str(e)}'
            }
    
    def ping_check(self, hostname: str) -> Dict[str, Any]:
        """Ping kontrolü"""
        try:
            result = subprocess.run(
                ['ping', '-c', '3', hostname],
                capture_output=True,
                text=True,
                timeout=15
            )
            
            # Ping istatistiklerini parse et
            lines = result.stdout.split('\n')
            stats_line = [line for line in lines if 'packet loss' in line]
            
            if stats_line:
                # "3 packets transmitted, 3 received, 0% packet loss, time 2002ms"
                parts = stats_line[0].split(',')
                loss_part = [p for p in parts if 'packet loss' in p][0]
                packet_loss = loss_part.strip().split('%')[0].split()[-1]
                
                return {
                    'success': True,
                    'packet_loss': f"{packet_loss}%",
                    'output': result.stdout,
                    'return_code': result.returncode
                }
            else:
                return {
                    'success': False,
                    'error': 'Could not parse ping output',
                    'output': result.stdout
                }
                
        except subprocess.TimeoutExpired:
            return {
                'success': False,
                'error': 'Ping timeout'
            }
        except Exception as e:
            return {
                'success': False,
                'error': f'Ping error: {str(e)}'
            }
    
    async def run_full_check(self) -> Dict[str, Any]:
        """Tam ağ kontrolü"""
        hostname = self.url.replace('https://', '').replace('http://', '').split('/')[0]
        
        check_result: Dict[str, Any] = {
            'timestamp': datetime.now().isoformat(),
            'url': self.url,
            'hostname': hostname,
        }
        
        # DNS kontrolü
        dns_result = await self.check_dns_resolution(hostname)
        check_result['dns'] = dns_result
        
        if not dns_result['success']:
            self.stats['dns_failures'] += 1
            logger.error(f"DNS resolution failed for {hostname}: {dns_result['error']}")
            return check_result
        
        # Ping kontrolü
        ping_result = self.ping_check(hostname)
        check_result['ping'] = ping_result
        
        # HTTP kontrolü
        http_result = await self.check_http_connectivity()
        check_result['http'] = http_result
        
        # İstatistikleri güncelle
        self.stats['total_checks'] += 1
        
        if http_result['success']:
            self.stats['successful_requests'] += 1
            logger.info(f"✅ Connection successful - Status: {http_result['status_code']}, "
                       f"Response time: {http_result['response_time_ms']}ms")
        else:
            self.stats['failed_requests'] += 1
            error_type = http_result.get('error_type', 'unknown')
            
            if error_type == 'timeout':
                self.stats['connection_timeouts'] += 1
            elif error_type == 'http':
                self.stats['http_errors'] += 1
                
            logger.error(f"❌ Connection failed: {http_result['error']}")
            
            # Hata durumunda traceroute çalıştır
            traceroute_result = self.check_traceroute(hostname)
            check_result['traceroute'] = traceroute_result
            
            if traceroute_result['success']:
                logger.info("Traceroute output:")
                for line in traceroute_result['output'].split('\n')[:10]:  # İlk 10 satır
                    if line.strip():
                        logger.info(f"  {line}")
        
        return check_result
    
    def print_stats(self):
        """İstatistikleri yazdır"""
        if self.stats['total_checks'] > 0:
            success_rate = (self.stats['successful_requests'] / self.stats['total_checks']) * 100
            
            print(f"\n📊 Network Monitor Statistics:")
            print(f"📅 Started: {self.stats['start_time']}")
            print(f"🔢 Total checks: {self.stats['total_checks']}")
            print(f"✅ Successful: {self.stats['successful_requests']} ({success_rate:.1f}%)")
            print(f"❌ Failed: {self.stats['failed_requests']}")
            print(f"🌐 DNS failures: {self.stats['dns_failures']}")
            print(f"⏱️ Timeouts: {self.stats['connection_timeouts']}")
            print(f"🚫 HTTP errors: {self.stats['http_errors']}")
            print("-" * 50)
    
    async def monitor(self, duration_minutes: Optional[int] = None):
        """Sürekli monitoring başlat"""
        logger.info(f"🚀 Starting network monitor for {self.url}")
        logger.info(f"⏱️ Check interval: {self.check_interval} seconds")
        
        if duration_minutes:
            logger.info(f"⏲️ Duration: {duration_minutes} minutes")
            end_time = time.time() + (duration_minutes * 60)
        else:
            logger.info("⏲️ Duration: Unlimited (Ctrl+C to stop)")
            end_time = None
        
        try:
            while True:
                if end_time and time.time() > end_time:
                    break
                    
                await self.run_full_check()
                
                # Her 10 kontrolde bir istatistikleri göster
                if self.stats['total_checks'] % 10 == 0:
                    self.print_stats()
                
                await asyncio.sleep(self.check_interval)
                
        except KeyboardInterrupt:
            logger.info("\n🛑 Monitoring stopped by user")
        finally:
            self.print_stats()
            logger.info("📝 Detailed logs saved to network_monitor.log")

async def main():
    """Ana fonksiyon"""
    url = "https://dinkal-poc-teams.yazge.aktekbilisim.com/"
    monitor = NetworkMonitor(url, check_interval=1)  # 30 saniyede bir kontrol
    
    # 60 dakika boyunca monitor et (None olursa sonsuz)
    await monitor.monitor(duration_minutes=60)

if __name__ == "__main__":
    asyncio.run(main())
