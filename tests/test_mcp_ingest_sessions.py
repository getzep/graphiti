import pytest

from scripts.mcp_ingest_sessions import (
    MCPClient,
    _validate_mcp_url,
    strip_graphiti_context,
    strip_ingestion_noise,
    strip_untrusted_metadata,
    subchunk_evidence,
)


def test_strip_untrusted_metadata():
    content = """
Hi there!

Conversation info (untrusted metadata):
```json
{
  "message_id": "123",
  "nested": "```python\\nfoo\\n```"
}
```

End of message.
"""
    expected = "Hi there!\n\nEnd of message."
    
    assert strip_untrusted_metadata(content) == expected

def test_strip_graphiti_context_wrapper_block():
    content = """
<graphiti-context>
## Graphiti Recall
- old memory item
</graphiti-context>

Actual user body.
"""
    stripped = strip_graphiti_context(content)
    assert '<graphiti-context>' not in stripped
    assert '</graphiti-context>' not in stripped
    assert 'Actual user body.' in stripped


def test_strip_graphiti_context_inline():
    """Inline form: both tags on the same line must be stripped."""
    content = '<graphiti-context>old memory item</graphiti-context>\n\nActual user body.'
    stripped = strip_graphiti_context(content)
    assert '<graphiti-context>' not in stripped
    assert '</graphiti-context>' not in stripped
    assert 'old memory item' not in stripped
    assert 'Actual user body.' in stripped


def test_strip_graphiti_context_inline_prefix_suffix():
    """Inline form with surrounding text: surrounding text must be preserved."""
    content = '[timestamp] <graphiti-context>memory</graphiti-context> rest of line\n'
    stripped = strip_graphiti_context(content)
    assert '<graphiti-context>' not in stripped
    assert 'memory' not in stripped
    assert '[timestamp]' in stripped
    assert 'rest of line' in stripped


def test_strip_graphiti_context_inline_case_insensitive():
    """Inline wrapper detection is case-insensitive."""
    content = '<GRAPHITI-CONTEXT>sensitive data</GRAPHITI-CONTEXT>\n\nBody.'
    stripped = strip_graphiti_context(content)
    assert 'sensitive data' not in stripped
    assert 'Body.' in stripped


def test_strip_ingestion_noise_removes_wrapper_and_metadata():
    content = """
<graphiti-context>
## Graphiti Recall
- previous memory snippet
</graphiti-context>

Sender (untrusted metadata):
```json
{"sender_id": "42"}
```

Conversation info (untrusted metadata):
```json
{"session": "abc"}
```

This is the actual message body.
"""
    assert strip_ingestion_noise(content) == "This is the actual message body."


def test_strip_metadata_multiple_blocks():
    content = """
Block 1.

Sender (untrusted metadata):
```json
{"user": "Alice"}
```

Block 2.

Conversation info:
```json
{"id": "456"}
```

Block 3.
"""
    expected = "Block 1.\n\nBlock 2.\n\nBlock 3."
    assert strip_untrusted_metadata(content) == expected

def test_strip_metadata_no_match():
    content = "Hello\\n\\nWorld"
    assert strip_untrusted_metadata(content) == content.strip()

def test_strip_metadata_unclosed_fence():
    # An adversary doesn't close the fence. It shouldn't hang, and shouldn't strip.
    content = """
Conversation info (untrusted metadata):
```json
{"malicious": "payload"
"""
    # Without the closing fence, it shouldn't strip it.
    assert strip_untrusted_metadata(content) == content.strip()

def test_subchunk_evidence_uses_strip():
    content = """
Conversation info:
```json
{"id": "test"}
```
Test message.
"""
    # max_chars=100 won't split "Test message."
    chunks = subchunk_evidence(content.strip(), "my_key", 100)
    assert len(chunks) == 1
    assert chunks[0][0] == "my_key"
    assert chunks[0][1] == "Test message."


def test_subchunk_evidence_strips_graphiti_context_wrapper():
    content = """
<graphiti-context>
## Graphiti Recall
- noisy summary
</graphiti-context>

Useful body content.
"""
    chunks = subchunk_evidence(content.strip(), "my_key", 100)
    assert len(chunks) == 1
    assert chunks[0][1] == "Useful body content."


# ---------------------------------------------------------------------------
# NEW: SSRF hardening — _validate_mcp_url
# ---------------------------------------------------------------------------


def test_validate_mcp_url_allows_localhost() -> None:
    """Localhost MCP URL is explicitly allowed (expected deployment target)."""
    assert _validate_mcp_url("http://localhost:8000/mcp") == "http://localhost:8000/mcp"


def test_validate_mcp_url_allows_private_ip() -> None:
    """Private RFC-1918 IPs are allowed (MCP server may run on local network)."""
    assert _validate_mcp_url("http://192.168.1.50:8000/mcp") == "http://192.168.1.50:8000/mcp"


def test_validate_mcp_url_allows_https() -> None:
    """HTTPS is always valid for MCP."""
    assert _validate_mcp_url("https://graphiti.internal/mcp") == "https://graphiti.internal/mcp"


def test_validate_mcp_url_rejects_non_http_scheme() -> None:
    """Non-http(s) schemes (file://, gopher://) must be rejected."""
    with pytest.raises(ValueError, match="http"):
        _validate_mcp_url("file:///etc/passwd")


def test_validate_mcp_url_rejects_gopher() -> None:
    with pytest.raises(ValueError, match="http"):
        _validate_mcp_url("gopher://localhost/evil")


def test_validate_mcp_url_rejects_embedded_credentials() -> None:
    """URLs with embedded user:pass credentials are rejected."""
    with pytest.raises(ValueError, match="credentials"):
        _validate_mcp_url("http://user:secret@localhost:8000/mcp")


def test_validate_mcp_url_rejects_link_local_cloud_metadata() -> None:
    """Cloud metadata IP 169.254.169.254 (link-local) is always blocked."""
    with pytest.raises(ValueError, match="link-local"):
        _validate_mcp_url("http://169.254.169.254/latest/meta-data/")


def test_validate_mcp_url_rejects_empty_netloc() -> None:
    """URL with no host is rejected."""
    with pytest.raises(ValueError, match="http"):
        _validate_mcp_url("http:///mcp")


def test_mcp_client_validates_url_on_init() -> None:
    """MCPClient raises ValueError on init when URL fails validation."""
    with pytest.raises(ValueError):
        MCPClient("file:///etc/passwd")


def test_mcp_client_accepts_localhost_url() -> None:
    """MCPClient accepts localhost MCP URL on init."""
    client = MCPClient("http://localhost:8000/mcp")
    assert client.url == "http://localhost:8000/mcp"


# ---------------------------------------------------------------------------
# NO-GO fixes: IPv6 parsing + query/fragment rejection in _validate_mcp_url
# ---------------------------------------------------------------------------


def test_validate_mcp_url_rejects_query_string() -> None:
    """MCP URLs with query parameters must be rejected.

    The docstring documents this requirement; this test ensures code enforces it.
    """
    with pytest.raises(ValueError, match="query"):
        _validate_mcp_url("http://localhost:8000/mcp?token=abc")


def test_validate_mcp_url_rejects_query_string_complex() -> None:
    """Reject complex query strings that could indicate injection attempts."""
    with pytest.raises(ValueError, match="query"):
        _validate_mcp_url("http://localhost:8000/mcp?foo=bar&baz=1")


def test_validate_mcp_url_rejects_fragment() -> None:
    """MCP URLs with URL fragments must be rejected.

    The docstring documents this requirement; this test ensures code enforces it.
    """
    with pytest.raises(ValueError, match="fragment"):
        _validate_mcp_url("http://localhost:8000/mcp#section")


def test_validate_mcp_url_allows_path_without_query_or_fragment() -> None:
    """URLs with a path component (but no query/fragment) are valid."""
    url = "http://localhost:8000/mcp/v2"
    assert _validate_mcp_url(url) == url


def test_validate_mcp_url_blocks_ipv6_link_local() -> None:
    """IPv6 link-local addresses (fe80::1) are always blocked in MCP URLs.

    The naive netloc.split(':')[0] approach fails to extract the IPv6 address
    and silently allows link-local endpoints.  parsed.hostname fixes this.
    """
    with pytest.raises(ValueError, match="link-local"):
        _validate_mcp_url("http://[fe80::1]:8000/mcp")


def test_validate_mcp_url_allows_ipv6_loopback() -> None:
    """IPv6 loopback [::1] is allowed — MCP server can legitimately run on ::1."""
    url = "http://[::1]:8000/mcp"
    assert _validate_mcp_url(url) == url


def test_validate_mcp_url_allows_ipv6_loopback_no_port() -> None:
    """IPv6 loopback [::1] without explicit port is allowed."""
    url = "http://[::1]/mcp"
    assert _validate_mcp_url(url) == url
