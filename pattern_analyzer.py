#!/usr/bin/env python3
"""
Pattern analysis script to detect load balancer issues
"""

import json
import subprocess
import time
from datetime import datetime
from collections import Counter

def analyze_failure_pattern():
    """Connectivity sonuçlarındaki pattern'i analiz et"""
    
    with open('connectivity_results.json', 'r') as f:
        data = json.load(f)
    
    results = data['results']
    
    print("🔍 FAILURE PATTERN ANALYSIS")
    print("=" * 50)
    
    # Sequential pattern analizi
    pattern = []
    for result in results:
        pattern.append('✅' if result['success'] else '❌')
    
    print(f"📊 Pattern (last 20): {''.join(pattern[-20:])}")
    
    # Consecutive failure/success groups
    groups = []
    current_group = [pattern[0]]
    
    for i in range(1, len(pattern)):
        if pattern[i] == pattern[i-1]:
            current_group.append(pattern[i])
        else:
            groups.append(current_group)
            current_group = [pattern[i]]
    groups.append(current_group)
    
    # Group statistics
    success_groups = [len(g) for g in groups if g[0] == '✅']
    failure_groups = [len(g) for g in groups if g[0] == '❌']
    
    print(f"🟢 Success groups: {success_groups}")
    print(f"🔴 Failure groups: {failure_groups}")
    
    # Average group sizes
    avg_success_group = sum(success_groups) / len(success_groups) if success_groups else 0
    avg_failure_group = sum(failure_groups) / len(failure_groups) if failure_groups else 0
    
    print(f"📈 Avg success streak: {avg_success_group:.1f}")
    print(f"📉 Avg failure streak: {avg_failure_group:.1f}")
    
    # Response time analysis for successful requests
    successful_results = [r for r in results if r['success']]
    response_times = [r['response_time_ms'] for r in successful_results]
    
    if response_times:
        avg_time = sum(response_times) / len(response_times)
        min_time = min(response_times)
        max_time = max(response_times)
        
        print(f"⏱️ Response times - Avg: {avg_time:.1f}ms, Min: {min_time:.1f}ms, Max: {max_time:.1f}ms")
        
        # Response time variation might indicate different backends
        time_variance = sum((t - avg_time) ** 2 for t in response_times) / len(response_times)
        print(f"📊 Response time variance: {time_variance:.1f}")
    
    # Pattern hypothesis
    print("\n🎯 PATTERN ANALYSIS:")
    
    if avg_failure_group >= 2 and avg_success_group >= 2:
        print("   🔄 LIKELY: Multiple backend instances with one failing")
        print("   💡 Action: Check backend instance health individually")
    elif avg_failure_group == 1 and avg_success_group >= 1:
        print("   ⚡ LIKELY: Rate limiting or connection throttling")
        print("   💡 Action: Check rate limits and connection pools")
    elif len(groups) > len(results) * 0.8:
        print("   🌐 LIKELY: Network instability or DNS issues")
        print("   💡 Action: Check network infrastructure")
    else:
        print("   ❓ UNCLEAR: Need more data or different analysis")
    
    return {
        'total_groups': len(groups),
        'success_groups': len(success_groups),
        'failure_groups': len(failure_groups),
        'avg_success_streak': avg_success_group,
        'avg_failure_streak': avg_failure_group
    }

def test_backend_affinity():
    """Backend affinity test - aynı session'da consistency var mı?"""
    print("\n🔗 BACKEND AFFINITY TEST")
    print("=" * 50)
    
    url = "https://dinkal-poc-teams.yazge.aktekbilisim.com/"
    
    # 10 rapid request - aynı connection kullanmaya çalış
    print("Sending 10 rapid requests to test session affinity...")
    
    results = []
    for i in range(10):
        try:
            start_time = time.time()
            result = subprocess.run([
                'curl', '-I', '-s', '-m', '3', '--keepalive-time', '60', url
            ], capture_output=True, text=True, timeout=5)
            
            timing = (time.time() - start_time) * 1000
            success = result.returncode == 0
            
            status = "✅" if success else "❌"
            print(f"   {i+1:2d}. {status} ({timing:6.1f}ms)")
            
            results.append(success)
            
            time.sleep(0.5)  # 500ms interval
            
        except Exception as e:
            print(f"   {i+1:2d}. ❌ Error: {e}")
            results.append(False)
    
    # Affinity analysis
    success_count = sum(results)
    if success_count == 10:
        print("   💚 RESULT: All requests successful - No load balancer issue")
    elif success_count == 0:
        print("   💔 RESULT: All requests failed - Service completely down")
    else:
        print(f"   ⚠️ RESULT: {success_count}/10 successful - CONFIRMS load balancer issue")
        
        # Pattern in rapid requests
        rapid_pattern = ''.join(['✅' if r else '❌' for r in results])
        print(f"   📊 Rapid pattern: {rapid_pattern}")

def check_dns_consistency():
    """DNS consistency check"""
    print("\n🌐 DNS CONSISTENCY CHECK")
    print("=" * 50)
    
    hostname = "dinkal-poc-teams.yazge.aktekbilisim.com"
    
    # Multiple DNS queries
    ips = []
    for i in range(5):
        try:
            result = subprocess.run(['dig', '+short', 'A', hostname], 
                                  capture_output=True, text=True, timeout=5)
            if result.stdout.strip():
                current_ips = result.stdout.strip().split('\n')
                ips.extend(current_ips)
                print(f"   Query {i+1}: {', '.join(current_ips)}")
            time.sleep(1)
        except Exception as e:
            print(f"   Query {i+1}: Error - {e}")
    
    unique_ips = list(set(ips))
    if len(unique_ips) == 1:
        print(f"   ✅ DNS consistent: Single IP {unique_ips[0]}")
    else:
        print(f"   ⚠️ DNS inconsistent: Multiple IPs {unique_ips}")
        print(f"   💡 This could indicate multiple backend instances")

def main():
    """Ana fonksiyon"""
    print("🔬 ADVANCED PATTERN ANALYSIS")
    print("=" * 50)
    print(f"⏰ {datetime.now().strftime('%H:%M:%S')}")
    print()
    
    # 1. Pattern analysis
    pattern_stats = analyze_failure_pattern()
    
    # 2. Backend affinity test
    test_backend_affinity()
    
    # 3. DNS consistency
    check_dns_consistency()
    
    print("\n" + "=" * 50)
    print("🎯 CONCLUSION & RECOMMENDATIONS:")
    print("=" * 50)
    
    if pattern_stats['avg_failure_streak'] >= 2:
        print("🔄 HIGH PROBABILITY: Multiple backend instances")
        print("   → Check Traefik backend configuration")
        print("   → Verify all backend instances are healthy")
        print("   → Review health check settings")
    else:
        print("⚡ HIGH PROBABILITY: Rate limiting or resource constraint")
        print("   → Check connection limits")
        print("   → Review resource usage")
        print("   → Analyze rate limiting rules")

if __name__ == "__main__":
    main()
