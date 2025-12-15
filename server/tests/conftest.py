import sys
from pathlib import Path

# Ensure `server/` is on `sys.path` so tests can import `graph_service.*` when running from repo root.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
