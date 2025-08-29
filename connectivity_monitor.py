#!/usr/bin/env python3
"""
Continuous monitoring script specifically for the intermittent connectivity issue
"""

import subprocess
import time
import json
from datetime import datetime
from collections import deque

class ConnectivityMonitor:
    def __init__(self, url: str, test_interval: int = 10):
        self.url = url
        self.hostname = url.replace('https://', '').replace('http://', '').split('/')[0]
        self.test_interval = test_interval
        self.results = deque(maxlen=100)  # Son 100 test sonucu
        
    def single_test(self) -> dict:
        """Tek HTTP testi"""
        try:
            start_time = time.time()
            result = subprocess.run([
                'curl', '-I', '-s', '-m', '5', '--connect-timeout', '3', self.url
            ], capture_output=True, text=True, timeout=8)
            
            timing = (time.time() - start_time) * 1000
            timestamp = datetime.now()
            
            if result.returncode == 0:
                first_line = result.stdout.split('\n')[0] if result.stdout else ''
                status_code = first_line.split()[1] if len(first_line.split()) > 1 else 'Unknown'
                
                return {
                    'timestamp': timestamp.isoformat(),
                    'success': True,
                    'status_code': status_code,
                    'response_time_ms': round(timing, 2),
                    'full_response': result.stdout
                }
            else:
                return {
                    'timestamp': timestamp.isoformat(),
                    'success': False,
                    'error': result.stderr or 'Connection failed',
                    'response_time_ms': round(timing, 2)
                }
                
        except Exception as e:
            return {
                'timestamp': datetime.now().isoformat(),
                'success': False,
                'error': f'Test exception: {str(e)}',
                'response_time_ms': None
            }
    
    def calculate_stats(self) -> dict:
        """İstatistikleri hesapla"""
        if not self.results:
            return {}
            
        total = len(self.results)
        successful = sum(1 for r in self.results if r['success'])
        failed = total - successful
        
        success_rate = (successful / total) * 100 if total > 0 else 0
        
        # Son 10 test
        recent_results = list(self.results)[-10:]
        recent_successful = sum(1 for r in recent_results if r['success'])
        recent_success_rate = (recent_successful / len(recent_results)) * 100 if recent_results else 0
        
        # Response time istatistikleri
        response_times = [r['response_time_ms'] for r in self.results if r['success'] and r['response_time_ms']]
        avg_response_time = sum(response_times) / len(response_times) if response_times else 0
        
        return {
            'total_tests': total,
            'successful': successful,
            'failed': failed,
            'success_rate': round(success_rate, 1),
            'recent_success_rate': round(recent_success_rate, 1),
            'avg_response_time_ms': round(avg_response_time, 2)
        }
    
    def print_status(self, result: dict, stats: dict):
        """Status yazdır"""
        timestamp = datetime.now().strftime('%H:%M:%S')
        
        if result['success']:
            status = f"✅ {result['status_code']} ({result['response_time_ms']}ms)"
        else:
            status = f"❌ FAILED - {result['error']}"
        
        print(f"[{timestamp}] {status}")
        
        # Her 10 testte bir özet göster
        if stats['total_tests'] % 10 == 0:
            print(f"   📊 Stats: {stats['successful']}/{stats['total_tests']} "
                  f"({stats['success_rate']}%) | Recent: {stats['recent_success_rate']}% | "
                  f"Avg: {stats['avg_response_time_ms']}ms")
            print("-" * 60)
    
    def save_results(self, filename: str = 'connectivity_results.json'):
        """Sonuçları dosyaya kaydet"""
        data = {
            'url': self.url,
            'test_interval': self.test_interval,
            'results': list(self.results),
            'stats': self.calculate_stats(),
            'generated_at': datetime.now().isoformat()
        }
        
        with open(filename, 'w') as f:
            json.dump(data, f, indent=2)
    
    def run_monitoring(self, duration_minutes: int = 30):
        """Monitoring başlat"""
        print(f"🚀 Starting connectivity monitoring for {self.url}")
        print(f"⏱️ Test interval: {self.test_interval} seconds")
        print(f"⏲️ Duration: {duration_minutes} minutes")
        print("=" * 60)
        
        end_time = time.time() + (duration_minutes * 60)
        
        try:
            while time.time() < end_time:
                # Test yap
                result = self.single_test()
                self.results.append(result)
                
                # İstatistikleri hesapla ve göster
                stats = self.calculate_stats()
                self.print_status(result, stats)
                
                # Sonuçları kaydet (her 20 testte bir)
                if stats['total_tests'] % 20 == 0:
                    self.save_results()
                
                # Bekle
                time.sleep(self.test_interval)
                
        except KeyboardInterrupt:
            print(f"\n🛑 Monitoring stopped by user")
        
        # Final istatistikler
        final_stats = self.calculate_stats()
        print(f"\n📊 FINAL STATISTICS:")
        print(f"   Total tests: {final_stats['total_tests']}")
        print(f"   Success rate: {final_stats['success_rate']}%")
        print(f"   Recent success rate: {final_stats['recent_success_rate']}%")
        print(f"   Average response time: {final_stats['avg_response_time_ms']}ms")
        
        # Final kaydet
        self.save_results()
        print(f"   Results saved to connectivity_results.json")

def main():
    url = "https://dinkal-poc-teams.yazge.aktekbilisim.com/"
    monitor = ConnectivityMonitor(url, test_interval=1)  # 10 saniyede bir test
    monitor.run_monitoring(duration_minutes=30)  # 30 dakika

if __name__ == "__main__":
    main()
