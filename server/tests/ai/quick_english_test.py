#!/usr/bin/env python3
import sys, os
sys.path.insert(0, '/c/pawelz-workspace/graphiti/server')

print("=== ANGIELSKIE TŁUMACZENIE PRZYKŁADU ===")

try:
    from graph_service.routers.ai.coreference_resolver import get_coreference_resolver
    
    resolver = get_coreference_resolver()
    
    # Tłumaczenie twojego polskiego przykładu na angielski
    polish_original = "poszedłem z jarkiem do kina. oglądaliśmy mecz tam. powiedział że mecz był ok"
    english_translation = "I went to the cinema with Jarek. We watched a match there. He said the match was good."
    
    print(f"Polish:  {polish_original}")
    print(f"English: {english_translation}")
    print()
    
    if resolver.is_available():
        # Test polskiego (dla porównania)
        print("🇵🇱 TESTING POLISH:")
        result_pl = resolver.model.predict([polish_original])
        clusters_pl = result_pl[0].get_clusters() if result_pl else []
        print(f"   Clusters: {len(clusters_pl)} -> {clusters_pl}")
        
        # Test angielskiego
        print("\n🇺🇸 TESTING ENGLISH:")
        result_en = resolver.model.predict([english_translation])
        clusters_en = result_en[0].get_clusters() if result_en else []
        print(f"   Clusters: {len(clusters_en)} -> {clusters_en}")
        
        print(f"\n📊 COMPARISON:")
        print(f"   Polish clusters:  {len(clusters_pl)}")
        print(f"   English clusters: {len(clusters_en)}")
        
        if len(clusters_en) > len(clusters_pl):
            print("   ✅ English translation works better!")
            for i, cluster in enumerate(clusters_en, 1):
                print(f"      Cluster {i}: {cluster}")
                if 'Jarek' in cluster and 'He' in cluster:
                    print(f"      🎉 SUCCESS: 'He' resolved to 'Jarek'!")
        elif len(clusters_en) == 0 and len(clusters_pl) == 0:
            print("   ⚠️  Neither language version works")
        else:
            print("   🤔 Unexpected result")
            
    else:
        print("❌ FastCoref not available")
        
except Exception as e:
    print(f"❌ Error: {e}")
    import traceback
    traceback.print_exc()
