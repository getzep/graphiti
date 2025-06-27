"""
Telemetry module for Graphiti.

This module provides anonymous usage analytics to help improve Graphiti.
"""

from .telemetry import capture_event, is_telemetry_enabled

__all__ = ['capture_event', 'is_telemetry_enabled']
