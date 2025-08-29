#!/usr/bin/env python3
"""
SSL Certificate Authority and Management Analysis
Determines who is providing and managing the SSL certificate
"""

import subprocess
import json
import re
from datetime import datetime

class SSLCertificateAnalyzer:
    def __init__(self, domain):
        self.domain = domain
        self.results = {}
    
    def get_certificate_details(self):
        """SSL sertifikası detaylarını al"""
        print("🔒 SSL CERTIFICATE ANALYSIS")
        print("=" * 50)
        
        try:
            # Get full certificate details
            cmd = [
                'openssl', 's_client', '-connect', f'{self.domain}:443',
                '-servername', self.domain, '-showcerts'
            ]
            
            result = subprocess.run(cmd, input='\n', capture_output=True, 
                                  text=True, timeout=15)
            
            if result.returncode == 0:
                # Parse certificate info
                cert_info = self.parse_certificate_output(result.stdout)
                self.results['certificate'] = cert_info
                
                print(f"📜 Certificate Subject: {cert_info.get('subject', 'N/A')}")
                print(f"🏢 Certificate Issuer: {cert_info.get('issuer', 'N/A')}")
                print(f"📅 Valid From: {cert_info.get('not_before', 'N/A')}")
                print(f"📅 Valid Until: {cert_info.get('not_after', 'N/A')}")
                print(f"🔐 Key Algorithm: {cert_info.get('signature_algorithm', 'N/A')}")
                print(f"🌐 SAN Domains: {', '.join(cert_info.get('san_domains', []))}")
                
            else:
                print(f"❌ Failed to get certificate: {result.stderr}")
                
        except Exception as e:
            print(f"❌ Certificate analysis error: {e}")
    
    def parse_certificate_output(self, output):
        """OpenSSL output'unu parse et"""
        cert_info = {}
        
        # Extract certificate text
        cert_match = re.search(r'-----BEGIN CERTIFICATE-----(.*?)-----END CERTIFICATE-----', 
                              output, re.DOTALL)
        
        if cert_match:
            # Get certificate details using openssl x509
            cert_data = f"-----BEGIN CERTIFICATE-----{cert_match.group(1)}-----END CERTIFICATE-----"
            
            try:
                # Parse with openssl x509
                proc = subprocess.run(['openssl', 'x509', '-noout', '-text'], 
                                    input=cert_data, capture_output=True, text=True)
                
                if proc.returncode == 0:
                    cert_text = proc.stdout
                    
                    # Parse subject
                    subject_match = re.search(r'Subject: (.+)', cert_text)
                    if subject_match:
                        cert_info['subject'] = subject_match.group(1).strip()
                    
                    # Parse issuer
                    issuer_match = re.search(r'Issuer: (.+)', cert_text)
                    if issuer_match:
                        cert_info['issuer'] = issuer_match.group(1).strip()
                    
                    # Parse validity dates
                    not_before_match = re.search(r'Not Before: (.+)', cert_text)
                    if not_before_match:
                        cert_info['not_before'] = not_before_match.group(1).strip()
                    
                    not_after_match = re.search(r'Not After : (.+)', cert_text)
                    if not_after_match:
                        cert_info['not_after'] = not_after_match.group(1).strip()
                    
                    # Parse signature algorithm
                    sig_alg_match = re.search(r'Signature Algorithm: (.+)', cert_text)
                    if sig_alg_match:
                        cert_info['signature_algorithm'] = sig_alg_match.group(1).strip()
                    
                    # Parse SAN domains
                    san_match = re.search(r'Subject Alternative Name:\s*\n\s*(.+)', cert_text)
                    if san_match:
                        san_line = san_match.group(1)
                        domains = re.findall(r'DNS:([^,\s]+)', san_line)
                        cert_info['san_domains'] = domains
                        
            except Exception as e:
                print(f"Warning: Could not parse certificate details: {e}")
        
        return cert_info
    
    def check_certificate_authority(self):
        """Sertifika otoritesini analiz et"""
        print(f"\n🏛️ CERTIFICATE AUTHORITY ANALYSIS")
        print("=" * 50)
        
        issuer = self.results.get('certificate', {}).get('issuer', '')
        
        if not issuer:
            print("❌ No issuer information available")
            return
        
        print(f"📋 Full Issuer: {issuer}")
        
        # Analyze issuer type
        if 'Let\'s Encrypt' in issuer:
            print("🔰 Certificate Type: Let's Encrypt (Free, Automated)")
            print("   → Likely managed by Traefik or ACME client")
            print("   → Auto-renewal enabled")
            self.results['ca_type'] = 'letsencrypt'
            
        elif any(ca in issuer for ca in ['DigiCert', 'Comodo', 'GlobalSign', 'GeoTrust']):
            print("🏢 Certificate Type: Commercial CA")
            print("   → Manually purchased and installed")
            print("   → Could be managed by domain owner or Traefik")
            self.results['ca_type'] = 'commercial'
            
        elif 'Cloudflare' in issuer:
            print("☁️ Certificate Type: Cloudflare")
            print("   → Cloudflare is providing SSL termination")
            print("   → Domain is behind Cloudflare proxy")
            self.results['ca_type'] = 'cloudflare'
            
        elif any(word in issuer.lower() for word in ['internal', 'private', 'self']):
            print("🏠 Certificate Type: Internal/Self-Signed")
            print("   → Private certificate authority")
            print("   → Enterprise internal CA")
            self.results['ca_type'] = 'internal'
            
        else:
            print("❓ Certificate Type: Unknown/Other")
            print("   → Need manual investigation")
            self.results['ca_type'] = 'unknown'
    
    def check_traefik_acme_indicators(self):
        """Traefik ACME yönetimi işaretlerini kontrol et"""
        print(f"\n🔄 TRAEFIK ACME MANAGEMENT INDICATORS")
        print("=" * 50)
        
        # Check for Traefik-specific headers or patterns
        try:
            # Get full HTTP response headers
            cmd = ['curl', '-I', '-s', f'https://{self.domain}/']
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=10)
            
            if result.returncode == 0:
                headers = result.stdout.lower()
                
                # Look for Traefik indicators
                traefik_indicators = []
                
                if 'traefik' in headers:
                    traefik_indicators.append("Traefik header found")
                
                if 'x-forwarded' in headers:
                    traefik_indicators.append("X-Forwarded headers (proxy)")
                
                if any(server in headers for server in ['traefik', 'envoy', 'nginx']):
                    traefik_indicators.append("Proxy server detected")
                
                if traefik_indicators:
                    print("✅ Traefik/Proxy indicators found:")
                    for indicator in traefik_indicators:
                        print(f"   → {indicator}")
                else:
                    print("❌ No obvious Traefik indicators in headers")
                    
        except Exception as e:
            print(f"❌ Could not check headers: {e}")
    
    def check_domain_vs_traefik_management(self):
        """Domain vs Traefik yönetimi analizi"""
        print(f"\n🎯 SSL MANAGEMENT ANALYSIS")
        print("=" * 50)
        
        ca_type = self.results.get('ca_type', 'unknown')
        certificate = self.results.get('certificate', {})
        
        print("📊 Analysis Results:")
        
        if ca_type == 'letsencrypt':
            print("✅ LIKELY TRAEFIK MANAGED:")
            print("   → Let's Encrypt certificates are typically auto-managed")
            print("   → Traefik has built-in ACME support")
            print("   → Certificate renewal is automated")
            print("   💡 Check Traefik ACME configuration")
            
        elif ca_type == 'cloudflare':
            print("☁️ CLOUDFLARE MANAGED:")
            print("   → SSL is terminated at Cloudflare edge")
            print("   → Domain DNS points to Cloudflare")
            print("   → Backend connection might be HTTP")
            print("   💡 Check Cloudflare SSL/TLS settings")
            
        elif ca_type == 'commercial':
            print("🏢 POSSIBLY DOMAIN MANAGED:")
            print("   → Commercial certificate manually installed")
            print("   → Could be uploaded to Traefik or web server")
            print("   → Manual renewal required")
            print("   💡 Check certificate installation location")
            
        else:
            print("❓ UNCLEAR MANAGEMENT:")
            print("   → Need more investigation")
            print("   → Check Traefik configuration")
            print("   → Check domain hosting setup")
        
        # SAN domains analysis
        san_domains = certificate.get('san_domains', [])
        if san_domains:
            print(f"\n🌐 SAN Domains ({len(san_domains)} total):")
            for domain in san_domains[:5]:  # Show first 5
                print(f"   → {domain}")
            if len(san_domains) > 5:
                print(f"   → ... and {len(san_domains) - 5} more")
                
            if len(san_domains) == 1 and san_domains[0] == self.domain:
                print("   💡 Single domain certificate - likely Traefik ACME")
            elif len(san_domains) > 1:
                print("   💡 Multi-domain certificate - could be manually managed")
    
    def run_full_analysis(self):
        """Tam SSL analizi"""
        print("🔬 SSL CERTIFICATE PROVIDER ANALYSIS")
        print("=" * 70)
        print(f"🌐 Domain: {self.domain}")
        print(f"⏰ {datetime.now().strftime('%H:%M:%S')}")
        print()
        
        # 1. Get certificate details
        self.get_certificate_details()
        
        # 2. Analyze certificate authority
        self.check_certificate_authority()
        
        # 3. Check Traefik indicators
        self.check_traefik_acme_indicators()
        
        # 4. Domain vs Traefik analysis
        self.check_domain_vs_traefik_management()
        
        # 5. Save results
        self.save_results()
    
    def save_results(self, filename='ssl_analysis.json'):
        """Sonuçları kaydet"""
        data = {
            'domain': self.domain,
            'analysis_timestamp': datetime.now().isoformat(),
            'results': self.results
        }
        
        with open(filename, 'w') as f:
            json.dump(data, f, indent=2, default=str)
        
        print(f"\n💾 Analysis saved to {filename}")

def main():
    domain = "dinkal-poc-teams.yazge.aktekbilisim.com"
    analyzer = SSLCertificateAnalyzer(domain)
    analyzer.run_full_analysis()

if __name__ == "__main__":
    main()
