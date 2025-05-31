print("Test starting...")

try:
    import asyncio
    print("✓ asyncio imported")
    
    import sys
    import os
    # Add the server directory to the path
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
    print("✓ path setup done")
    
    from graph_service.routers.ai.coreference_resolver import get_coreference_resolver
    print("✓ coreference_resolver imported")
    
    resolver = get_coreference_resolver()
    print(f"✓ resolver created, available: {resolver.is_available()}")
    
    print("✅ All imports successful!")
    
except Exception as e:
    print(f"❌ Error: {e}")
    import traceback
    traceback.print_exc()
