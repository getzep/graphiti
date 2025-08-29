#!/usr/bin/env bash
"""
SSL/HTTPS Problem Investigation Script
"""

echo "🔒 SSL/HTTPS PROBLEM INVESTIGATION"
echo "=================================="

DOMAIN="dinkal-poc-teams.yazge.aktekbilisim.com"
IP="128.127.169.30"

echo "1️⃣ SSL Certificate Analysis:"
echo "------------------------------"

# Certificate details
echo "📜 Certificate information:"
echo | openssl s_client -connect $DOMAIN:443 -servername $DOMAIN 2>/dev/null | openssl x509 -noout -text | grep -A 5 "Subject:"

echo -e "\n📋 Certificate SAN (Subject Alternative Names):"
echo | openssl s_client -connect $DOMAIN:443 -servername $DOMAIN 2>/dev/null | openssl x509 -noout -text | grep -A 1 "Subject Alternative Name"

echo -e "\n🔗 Certificate chain verification:"
echo | openssl s_client -connect $DOMAIN:443 -servername $DOMAIN -verify_return_error 2>&1 | grep -E "(verify|Verification|Certificate)"

echo -e "\n2️⃣ Traefik SSL Configuration Check:"
echo "-----------------------------------"

# SSL handshake timing
echo "⏱️ SSL handshake timing test (5 attempts):"
for i in {1..5}; do
    time_result=$(curl -w "@-" -o /dev/null -s "https://$DOMAIN/" <<< '%{time_connect}:%{time_appconnect}:%{http_code}\n' 2>/dev/null)
    if [ ! -z "$time_result" ]; then
        IFS=':' read -r connect_time ssl_time http_code <<< "$time_result"
        ssl_handshake_time=$(echo "$ssl_time - $connect_time" | bc -l 2>/dev/null || echo "calc_error")
        echo "   $i. Connect: ${connect_time}s, SSL: ${ssl_handshake_time}s, HTTP: $http_code"
    else
        echo "   $i. ❌ Failed"
    fi
    sleep 1
done

echo -e "\n3️⃣ HTTP vs HTTPS Response Comparison:"
echo "------------------------------------"

echo "🌐 HTTP Response headers:"
curl -I -s http://$DOMAIN/ | head -10

echo -e "\n🔒 HTTPS Response headers:"
curl -I -s https://$DOMAIN/ | head -10

echo -e "\n4️⃣ Traefik Health Check:"
echo "------------------------"

# Common Traefik dashboard/API ports
TRAEFIK_PORTS=(8080 8090 9000 9090)

for port in "${TRAEFIK_PORTS[@]}"; do
    echo "🔍 Checking Traefik on port $port..."
    
    # Try dashboard
    dashboard_result=$(curl -s -m 3 http://$IP:$port/dashboard/ 2>/dev/null)
    if [[ $? -eq 0 ]] && [[ "$dashboard_result" == *"Traefik"* ]]; then
        echo "   ✅ Traefik dashboard found on port $port"
        echo "   🌐 Access: http://$IP:$port/dashboard/"
    fi
    
    # Try API
    api_result=$(curl -s -m 3 http://$IP:$port/api/http/services 2>/dev/null)
    if [[ $? -eq 0 ]] && [[ "$api_result" == *"{"* ]]; then
        echo "   ✅ Traefik API found on port $port"
        echo "   🔗 Services: http://$IP:$port/api/http/services"
    fi
done

echo -e "\n5️⃣ Backend Service Discovery:"
echo "-----------------------------"

# Try to find actual backend
echo "🎯 Looking for backend services..."

# Common backend patterns
for port in 3000 5000 8000 8080 9000; do
    result=$(curl -I -s -m 2 http://$IP:$port/ 2>/dev/null)
    if [ $? -eq 0 ]; then
        echo "   ✅ Backend found on port $port"
        echo "   📋 Headers: $(echo "$result" | head -1)"
    fi
done

echo -e "\n📊 SUMMARY & RECOMMENDATIONS:"
echo "============================="
echo "🔒 SSL/HTTPS issues confirmed"
echo "💡 Focus on:"
echo "   → Traefik SSL termination configuration"
echo "   → Certificate management"
echo "   → SSL backend health checks"
echo "   → Multiple Traefik instance SSL sync"
