"""
Telemetry client for Graphiti.

Collects anonymous usage statistics to help improve the product.
"""

import contextlib
import os
import platform
import sys
import uuid
from pathlib import Path
from typing import Any

# PostHog configuration
# Note: This is a public API key intended for client-side use and safe to commit
# PostHog public keys are designed to be exposed in client applications
POSTHOG_API_KEY = 'phc_UG6EcfDbuXz92neb3rMlQFDY0csxgMqRcIPWESqnSmo'
POSTHOG_HOST = 'https://us.i.posthog.com'

# Environment variable to control telemetry
TELEMETRY_ENV_VAR = 'GRAPHITI_TELEMETRY_ENABLED'

# Cache directory for anonymous ID
CACHE_DIR = Path.home() / '.cache' / 'graphiti'
ANON_ID_FILE = CACHE_DIR / 'telemetry_anon_id'


def is_telemetry_enabled() -> bool:
    """Check if telemetry is enabled."""
    # Disable during pytest runs
    if 'pytest' in sys.modules:
        return False

    # Check environment variable (default: enabled)
    env_value = os.environ.get(TELEMETRY_ENV_VAR, 'true').lower()
    return env_value in ('true', '1', 'yes', 'on')


def get_anonymous_id() -> str:
    """Get or create anonymous user ID."""
    try:
        # Create cache directory if it doesn't exist
        CACHE_DIR.mkdir(parents=True, exist_ok=True)

        # Try to read existing ID
        if ANON_ID_FILE.exists():
            try:
                return ANON_ID_FILE.read_text().strip()
            except Exception:
                pass

        # Generate new ID
        anon_id = str(uuid.uuid4())

        # Save to file
        with contextlib.suppress(Exception):
            ANON_ID_FILE.write_text(anon_id)

        return anon_id
    except Exception:
        return 'UNKNOWN'


def get_graphiti_version() -> str:
    """Get Graphiti version."""
    try:
        # Try to get version from package metadata
        import importlib.metadata

        return importlib.metadata.version('graphiti-core')
    except Exception:
        return 'unknown'


def initialize_posthog():
    """Initialize PostHog client."""
    try:
        import posthog

        posthog.api_key = POSTHOG_API_KEY
        posthog.host = POSTHOG_HOST
        return posthog
    except ImportError:
        # PostHog not installed, silently disable telemetry
        return None
    except Exception:
        # Any other error, silently disable telemetry
        return None


def capture_event(event_name: str, properties: dict[str, Any] | None = None) -> None:
    """Capture a telemetry event."""
    if not is_telemetry_enabled():
        return

    try:
        posthog_client = initialize_posthog()
        if posthog_client is None:
            return

        # Get anonymous ID
        user_id = get_anonymous_id()

        # Prepare event properties
        event_properties = {
            '$process_person_profile': False,
            'graphiti_version': get_graphiti_version(),
            'architecture': platform.machine(),
            **(properties or {}),
        }

        # Capture the event
        posthog_client.capture(distinct_id=user_id, event=event_name, properties=event_properties)
    except Exception:
        # Silently handle all telemetry errors to avoid disrupting the main application
        pass
