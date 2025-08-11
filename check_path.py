import sys
import os

print("\n--- Python Executable ---")
print(sys.executable)

print("\n--- sys.path (Module Search Paths) ---")
for path in sys.path:
    print(path)

print("\n--- Attempting to import google.generativeai ---")
try:
    import google.generativeai
    print("\n[SUCCESS] Successfully imported google.generativeai")
    print("Location:", google.generativeai.__file__)
except ModuleNotFoundError:
    print("\n[FAILURE] ModuleNotFoundError: Could not find google.generativeai.")
except Exception as e:
    print(f"\n[FAILURE] An unexpected error occurred: {e}")