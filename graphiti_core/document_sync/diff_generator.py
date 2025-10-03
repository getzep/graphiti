"""Utilities for content hashing and diff generation."""

import difflib
import hashlib


def compute_content_hash(content: str) -> str:
    """Generate SHA256 hash of content with 'sha256:' prefix.

    Args:
        content: The content to hash

    Returns:
        Hash string in format 'sha256:hexdigest'
    """
    hash_obj = hashlib.sha256(content.encode('utf-8'))
    return f'sha256:{hash_obj.hexdigest()}'


def generate_unified_diff(
    old_content: str,
    new_content: str,
    uri: str,
    context_lines: int = 100,
) -> str:
    """Generate unified diff with full context for LLM summarization.

    Args:
        old_content: Previous version of the document
        new_content: New version of the document
        uri: Document URI for diff header
        context_lines: Number of context lines (default: 100 for full document structure)

    Returns:
        Unified diff string for summarizer LLM
    """
    # Split content into lines for difflib
    old_lines = old_content.splitlines(keepends=True)
    new_lines = new_content.splitlines(keepends=True)

    # Generate unified diff with context lines
    diff_lines = difflib.unified_diff(
        old_lines,
        new_lines,
        fromfile=f'{uri} (previous)',
        tofile=f'{uri} (current)',
        n=context_lines,
    )

    # Join diff lines - no instruction header needed, summarizer will process
    diff_content = ''.join(diff_lines)

    return diff_content
