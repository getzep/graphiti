from __future__ import annotations

import importlib
import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

om_compressor = importlib.import_module("scripts.om_compressor")


def test_default_ontology_config_resolution_is_repo_relative(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.delenv("OM_ONTOLOGY_CONFIG_PATH", raising=False)
    monkeypatch.chdir(tmp_path)

    resolved = om_compressor._resolve_ontology_config_path(None)

    assert resolved == om_compressor.REPO_ROOT / om_compressor.DEFAULT_ONTOLOGY_CONFIG_REL


def test_ontology_config_env_override_resolves_relative_to_repo(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("OM_ONTOLOGY_CONFIG_PATH", "config/extraction_ontologies.yaml")

    resolved = om_compressor._resolve_ontology_config_path(None)

    assert resolved == om_compressor.REPO_ROOT / "config/extraction_ontologies.yaml"


def test_lock_path_default_not_legacy_tmp_file(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("OM_COMPRESSOR_LOCK_PATH", raising=False)
    monkeypatch.delenv("XDG_RUNTIME_DIR", raising=False)

    lock_path = om_compressor._resolve_lock_path()

    assert lock_path.name == om_compressor.DEFAULT_LOCK_FILENAME
    assert str(lock_path) != "/tmp/om_graph_write.lock"


def test_lock_path_env_override_is_honored(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("OM_COMPRESSOR_LOCK_PATH", "state/custom/om.lock")

    lock_path = om_compressor._resolve_lock_path()

    assert lock_path == om_compressor.REPO_ROOT / "state/custom/om.lock"


# ---------------------------------------------------------------------------
# OM-1: metadata stripping in om_compressor path
# ---------------------------------------------------------------------------


def test_strip_untrusted_metadata_removes_contamination_block() -> None:
    """Contaminated 'Conversation info (untrusted metadata)' JSON block is stripped."""
    content = (
        "Useful message content.\n\n"
        "Conversation info (untrusted metadata):\n"
        "```json\n"
        '{"message_id": "12345", "session_id": "abc", "source": "telegram"}\n'
        "```\n"
        "More content after block."
    )
    result = om_compressor.strip_untrusted_metadata(content)
    assert "Conversation info" not in result
    assert "message_id" not in result
    assert "session_id" not in result
    assert "Useful message content." in result
    assert "More content after block." in result


def test_strip_untrusted_metadata_clean_passthrough() -> None:
    """Content without metadata blocks passes through unchanged."""
    content = "Clean message with no metadata blocks. Just plain text."
    assert om_compressor.strip_untrusted_metadata(content) == content


def test_strip_untrusted_metadata_empty_string_safe() -> None:
    """Empty string returns empty string (no exception)."""
    assert om_compressor.strip_untrusted_metadata("") == ""


def test_strip_untrusted_metadata_none_passthrough() -> None:
    """None-equivalent (falsy) content returns the original value without crash."""
    # The function guards `if not content: return content`
    # so passing None would need the caller to coerce; test empty str only.
    assert om_compressor.strip_untrusted_metadata("") == ""


def test_strip_untrusted_metadata_all_four_prefixes() -> None:
    """All four untrusted-metadata prefixes are stripped."""
    prefixes = [
        "Conversation info:",
        "Sender (untrusted metadata):",
        "Replied message (untrusted, for context):",
        "Conversation info (untrusted metadata):",
    ]
    for prefix in prefixes:
        content = f"{prefix}\n```json\n{{\"key\": \"value\"}}\n```\nKeep this."
        result = om_compressor.strip_untrusted_metadata(content)
        assert prefix not in result, f"Prefix {prefix!r} was not stripped"
        assert "Keep this." in result, f"Non-metadata content was lost for prefix {prefix!r}"


def test_strip_untrusted_metadata_multiblock() -> None:
    """Multiple metadata blocks in one message are all stripped."""
    content = (
        "Lead text.\n\n"
        "Conversation info:\n```json\n{\"a\": 1}\n```\n"
        "Middle text.\n\n"
        "Sender (untrusted metadata):\n```json\n{\"b\": 2}\n```\n"
        "Tail text."
    )
    result = om_compressor.strip_untrusted_metadata(content)
    assert "Conversation info" not in result
    assert "Sender (untrusted metadata)" not in result
    assert "Lead text." in result
    assert "Middle text." in result
    assert "Tail text." in result


def test_strip_untrusted_metadata_strips_function_exists_on_module() -> None:
    """strip_untrusted_metadata is exported from om_compressor module."""
    assert callable(om_compressor.strip_untrusted_metadata)


# ---------------------------------------------------------------------------
# Extractor-path verification: OM_EXTRACTOR_PATH event
# ---------------------------------------------------------------------------


def _make_messages(n: int = 1) -> list:
    return [
        om_compressor.MessageRow(
            message_id=f"msg{i}",
            source_session_id="sess1",
            content=f"test content number {i}",
            created_at="2026-02-28T00:00:00Z",
            content_embedding=[],
            om_extract_attempts=0,
        )
        for i in range(n)
    ]


def _make_cfg(model_id: str = "gpt-5.1-codex-mini") -> om_compressor.ExtractorConfig:
    return om_compressor.ExtractorConfig(
        schema_version="v1",
        prompt_template="OM_PROMPT_TEMPLATE_V1",
        model_id=model_id,
        extractor_version="abc123",
    )


def test_extractor_path_event_emitted(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture
) -> None:
    """OM_EXTRACTOR_PATH event is emitted by _extract_items (model path, mocked LLM)."""
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test-key")
    monkeypatch.delenv("OM_EXTRACTOR_STRICT", raising=False)
    monkeypatch.delenv("OM_EXTRACTOR_MODE", raising=False)

    fake_chunk = om_compressor.ExtractedChunk(
        nodes=[
            om_compressor.ExtractionNode(
                node_id="n1",
                node_type="WorldState",
                semantic_domain="sessions_main",
                content="test fact",
                urgency_score=3,
                source_session_id="sess1",
                source_message_ids=["msg0"],
            )
        ],
        edges=[],
    )
    with patch("scripts.om_compressor._call_llm_extract", return_value=fake_chunk):
        om_compressor._extract_items(_make_messages(), _make_cfg())

    captured = capsys.readouterr()
    events = [json.loads(ln) for ln in captured.out.splitlines() if ln.strip() and '"event"' in ln]
    path_events = [e for e in events if e.get("event") == "OM_EXTRACTOR_PATH"]

    assert len(path_events) >= 1, "Expected at least one OM_EXTRACTOR_PATH event"
    evt = path_events[0]
    assert "extractor_mode" in evt, "extractor_mode field missing"
    assert evt["extractor_mode"] == "model", f"Expected model path, got: {evt['extractor_mode']!r}"
    assert "model_id" in evt, "model_id field missing"
    assert evt["model_id"] == "gpt-5.1-codex-mini"
    assert "strict_mode" in evt, "strict_mode field missing"


def test_extractor_path_event_fallback_has_reason(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture
) -> None:
    """Permissive fallback path emits OM_EXTRACTOR_PATH with a non-empty reason field."""
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("OM_EXTRACTOR_API_KEY", raising=False)
    # Permissive mode required — fallback is only allowed when explicitly opted in
    monkeypatch.setenv("OM_EXTRACTOR_STRICT", "false")

    om_compressor._extract_items(_make_messages(), _make_cfg())

    captured = capsys.readouterr()
    events = [json.loads(ln) for ln in captured.out.splitlines() if ln.strip() and '"event"' in ln]
    path_events = [e for e in events if e.get("event") == "OM_EXTRACTOR_PATH"]

    assert len(path_events) >= 1
    evt = path_events[0]
    assert evt["extractor_mode"] == "fallback"
    assert "reason" in evt, "reason field missing from fallback event"
    assert evt["reason"], "reason must be non-empty"


def test_extractor_path_event_model_id_propagated(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture
) -> None:
    """model_id in OM_EXTRACTOR_PATH matches the ExtractorConfig.model_id (permissive, no key)."""
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("OM_EXTRACTOR_API_KEY", raising=False)
    monkeypatch.setenv("OM_EXTRACTOR_STRICT", "false")

    custom_model = "my-custom-model-xyz"
    om_compressor._extract_items(_make_messages(), _make_cfg(model_id=custom_model))

    captured = capsys.readouterr()
    events = [json.loads(ln) for ln in captured.out.splitlines() if ln.strip() and '"event"' in ln]
    path_events = [e for e in events if e.get("event") == "OM_EXTRACTOR_PATH"]

    assert len(path_events) >= 1
    assert path_events[0]["model_id"] == custom_model


def test_is_model_client_available_false_when_no_keys(monkeypatch: pytest.MonkeyPatch) -> None:
    """_is_model_client_available returns False when no API keys are set."""
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("OM_EXTRACTOR_API_KEY", raising=False)
    assert om_compressor._is_model_client_available() is False


def test_is_model_client_available_true_with_openai_key(monkeypatch: pytest.MonkeyPatch) -> None:
    """_is_model_client_available returns True when OPENAI_API_KEY is set."""
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test-key")
    assert om_compressor._is_model_client_available() is True


def test_is_model_client_available_true_with_om_extractor_key(monkeypatch: pytest.MonkeyPatch) -> None:
    """_is_model_client_available returns True when OM_EXTRACTOR_API_KEY is set."""
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.setenv("OM_EXTRACTOR_API_KEY", "sk-om-test-key")
    assert om_compressor._is_model_client_available() is True


# ---------------------------------------------------------------------------


def test_relation_type_validation_and_interpolation_guard() -> None:
    assert om_compressor._validated_relation_type_for_cypher("motivates") == "MOTIVATES"

    edge = om_compressor.ExtractionEdge(
        source_node_id="n1",
        target_node_id="n2",
        relation_type="MOTIVATES",
    )
    assert om_compressor._assert_relation_type_safe_for_interpolation("MOTIVATES", edge) == "MOTIVATES"

    with pytest.raises(om_compressor.OMCompressorError):
        om_compressor._validated_relation_type_for_cypher("MOTIVATES]->(x) DETACH DELETE x //")

    with pytest.raises(om_compressor.OMCompressorError):
        om_compressor._assert_relation_type_safe_for_interpolation("BAD-TOKEN", edge)


# ---------------------------------------------------------------------------
# OM-2: Strict mode — fail-close extractor behaviour
# ---------------------------------------------------------------------------


def test_is_extractor_strict_defaults_true(monkeypatch: pytest.MonkeyPatch) -> None:
    """_is_extractor_strict() returns True when no env vars are set (fail-close default)."""
    monkeypatch.delenv("OM_EXTRACTOR_STRICT", raising=False)
    monkeypatch.delenv("OM_EXTRACTOR_MODE", raising=False)
    assert om_compressor._is_extractor_strict() is True


def test_is_extractor_strict_false_with_env_false(monkeypatch: pytest.MonkeyPatch) -> None:
    """OM_EXTRACTOR_STRICT=false disables strict mode."""
    monkeypatch.setenv("OM_EXTRACTOR_STRICT", "false")
    monkeypatch.delenv("OM_EXTRACTOR_MODE", raising=False)
    assert om_compressor._is_extractor_strict() is False


def test_is_extractor_strict_false_with_env_zero(monkeypatch: pytest.MonkeyPatch) -> None:
    """OM_EXTRACTOR_STRICT=0 disables strict mode."""
    monkeypatch.setenv("OM_EXTRACTOR_STRICT", "0")
    monkeypatch.delenv("OM_EXTRACTOR_MODE", raising=False)
    assert om_compressor._is_extractor_strict() is False


def test_is_extractor_strict_false_with_mode_permissive(monkeypatch: pytest.MonkeyPatch) -> None:
    """OM_EXTRACTOR_MODE=permissive disables strict mode (takes priority over STRICT env)."""
    monkeypatch.setenv("OM_EXTRACTOR_MODE", "permissive")
    monkeypatch.delenv("OM_EXTRACTOR_STRICT", raising=False)
    assert om_compressor._is_extractor_strict() is False


def test_is_extractor_strict_false_with_mode_debug(monkeypatch: pytest.MonkeyPatch) -> None:
    """OM_EXTRACTOR_MODE=debug disables strict mode."""
    monkeypatch.setenv("OM_EXTRACTOR_MODE", "debug")
    monkeypatch.delenv("OM_EXTRACTOR_STRICT", raising=False)
    assert om_compressor._is_extractor_strict() is False


def test_is_extractor_strict_true_with_mode_model_only(monkeypatch: pytest.MonkeyPatch) -> None:
    """Unknown OM_EXTRACTOR_MODE values preserve strict mode."""
    monkeypatch.setenv("OM_EXTRACTOR_MODE", "model_only")
    monkeypatch.delenv("OM_EXTRACTOR_STRICT", raising=False)
    assert om_compressor._is_extractor_strict() is True


# ---------------------------------------------------------------------------
# OM-2: Strict mode — _extract_items blocks fallback on no API key
# ---------------------------------------------------------------------------


def test_strict_mode_raises_on_no_api_key(monkeypatch: pytest.MonkeyPatch) -> None:
    """In strict mode (default), _extract_items raises when no API key is configured."""
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("OM_EXTRACTOR_API_KEY", raising=False)
    monkeypatch.delenv("OM_EXTRACTOR_STRICT", raising=False)
    monkeypatch.delenv("OM_EXTRACTOR_MODE", raising=False)

    with pytest.raises(om_compressor.OMExtractorStrictModeError):
        om_compressor._extract_items(_make_messages(), _make_cfg())


def test_strict_mode_raises_on_model_call_failure(monkeypatch: pytest.MonkeyPatch) -> None:
    """In strict mode, _extract_items raises when _call_llm_extract raises."""
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test-key")
    monkeypatch.delenv("OM_EXTRACTOR_STRICT", raising=False)
    monkeypatch.delenv("OM_EXTRACTOR_MODE", raising=False)

    with patch(
        "scripts.om_compressor._call_llm_extract",
        side_effect=om_compressor.OMCompressorError("simulated LLM timeout"),
    ), pytest.raises(om_compressor.OMExtractorStrictModeError):
        om_compressor._extract_items(_make_messages(), _make_cfg())


def test_strict_mode_emits_strict_block_event_on_no_key(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture
) -> None:
    """Strict mode emits OM_EXTRACTOR_STRICT_BLOCK event before raising."""
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("OM_EXTRACTOR_API_KEY", raising=False)
    monkeypatch.delenv("OM_EXTRACTOR_STRICT", raising=False)
    monkeypatch.delenv("OM_EXTRACTOR_MODE", raising=False)

    with pytest.raises(om_compressor.OMExtractorStrictModeError):
        om_compressor._extract_items(_make_messages(), _make_cfg())

    captured = capsys.readouterr()
    events = [json.loads(ln) for ln in captured.out.splitlines() if ln.strip() and '"event"' in ln]
    block_events = [e for e in events if e.get("event") == "OM_EXTRACTOR_STRICT_BLOCK"]

    assert len(block_events) >= 1, "OM_EXTRACTOR_STRICT_BLOCK event must be emitted"
    evt = block_events[0]
    assert evt.get("strict_mode") is True
    assert "reason" in evt
    assert evt["reason"] == "no_model_client"


def test_strict_mode_emits_strict_block_event_on_model_error(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture
) -> None:
    """Strict mode emits OM_EXTRACTOR_STRICT_BLOCK when model call fails."""
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test-key")
    monkeypatch.delenv("OM_EXTRACTOR_STRICT", raising=False)
    monkeypatch.delenv("OM_EXTRACTOR_MODE", raising=False)

    with patch(
        "scripts.om_compressor._call_llm_extract",
        side_effect=om_compressor.OMCompressorError("connection refused"),
    ), pytest.raises(om_compressor.OMExtractorStrictModeError):
        om_compressor._extract_items(_make_messages(), _make_cfg())

    captured = capsys.readouterr()
    events = [json.loads(ln) for ln in captured.out.splitlines() if ln.strip() and '"event"' in ln]
    block_events = [e for e in events if e.get("event") == "OM_EXTRACTOR_STRICT_BLOCK"]

    assert len(block_events) >= 1
    assert "connection refused" in block_events[0]["reason"]


# ---------------------------------------------------------------------------
# OM-2: Permissive mode — _extract_items allows fallback with warning
# ---------------------------------------------------------------------------


def test_permissive_mode_allows_fallback_on_no_key(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture
) -> None:
    """In permissive mode, _extract_items falls back to rules when no API key (no raise)."""
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("OM_EXTRACTOR_API_KEY", raising=False)
    monkeypatch.setenv("OM_EXTRACTOR_STRICT", "false")

    result = om_compressor._extract_items(_make_messages(3), _make_cfg())

    # Should not raise; should return a valid (rule-based) chunk
    assert isinstance(result, om_compressor.ExtractedChunk)
    # Rule-based path produces nodes but no ontology edges
    assert isinstance(result.nodes, list)
    assert isinstance(result.edges, list)

    captured = capsys.readouterr()
    events = [json.loads(ln) for ln in captured.out.splitlines() if ln.strip() and '"event"' in ln]
    path_events = [e for e in events if e.get("event") == "OM_EXTRACTOR_PATH"]
    assert any(e.get("extractor_mode") == "fallback" for e in path_events)


def test_permissive_mode_allows_fallback_on_model_error(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture
) -> None:
    """In permissive mode, _extract_items falls back to rules when model call fails."""
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test-key")
    monkeypatch.setenv("OM_EXTRACTOR_STRICT", "false")

    with patch(
        "scripts.om_compressor._call_llm_extract",
        side_effect=om_compressor.OMCompressorError("simulated timeout"),
    ):
        result = om_compressor._extract_items(_make_messages(2), _make_cfg())

    assert isinstance(result, om_compressor.ExtractedChunk)
    assert isinstance(result.nodes, list)

    captured = capsys.readouterr()
    events = [json.loads(ln) for ln in captured.out.splitlines() if ln.strip() and '"event"' in ln]
    permissive_events = [e for e in events if e.get("event") == "OM_EXTRACTOR_PERMISSIVE_FALLBACK"]
    assert len(permissive_events) >= 1, "OM_EXTRACTOR_PERMISSIVE_FALLBACK event must be emitted"
    assert permissive_events[0].get("warning") == "PERMISSIVE_MODE_FALLBACK"


def test_permissive_mode_via_mode_env(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """OM_EXTRACTOR_MODE=permissive enables permissive mode (no raise on no key)."""
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("OM_EXTRACTOR_API_KEY", raising=False)
    monkeypatch.setenv("OM_EXTRACTOR_MODE", "permissive")
    monkeypatch.delenv("OM_EXTRACTOR_STRICT", raising=False)

    # Must not raise
    result = om_compressor._extract_items(_make_messages(), _make_cfg())
    assert isinstance(result, om_compressor.ExtractedChunk)


# ---------------------------------------------------------------------------
# OM-2: Model path — non-empty edges emitted when LLM returns edges
# ---------------------------------------------------------------------------


def _fake_chunk_with_edges() -> om_compressor.ExtractedChunk:
    """Build a fake ExtractedChunk with two nodes and one MOTIVATES edge."""
    n1 = om_compressor.ExtractionNode(
        node_id="node-a",
        node_type="Commitment",
        semantic_domain="sessions_main",
        content="Yuan prefers meetings after 11am",
        urgency_score=4,
        source_session_id="sess1",
        source_message_ids=["msg0"],
    )
    n2 = om_compressor.ExtractionNode(
        node_id="node-b",
        node_type="OperationalRule",
        semantic_domain="sessions_main",
        content="No calls before 10:30am",
        urgency_score=4,
        source_session_id="sess1",
        source_message_ids=["msg1"],
    )
    edge = om_compressor.ExtractionEdge(
        source_node_id="node-a",
        target_node_id="node-b",
        relation_type="MOTIVATES",
    )
    return om_compressor.ExtractedChunk(nodes=[n1, n2], edges=[edge])


def test_model_path_emits_non_empty_edges(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture
) -> None:
    """Model path can emit non-empty edges; OM_EXTRACTOR_PATH reflects edge count."""
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test-key")
    monkeypatch.delenv("OM_EXTRACTOR_STRICT", raising=False)
    monkeypatch.delenv("OM_EXTRACTOR_MODE", raising=False)

    fake_chunk = _fake_chunk_with_edges()
    with patch("scripts.om_compressor._call_llm_extract", return_value=fake_chunk):
        result = om_compressor._extract_items(_make_messages(2), _make_cfg())

    assert len(result.edges) >= 1, "Model path must produce at least one edge"
    assert result.edges[0].relation_type == "MOTIVATES"

    captured = capsys.readouterr()
    events = [json.loads(ln) for ln in captured.out.splitlines() if ln.strip() and '"event"' in ln]
    path_events = [e for e in events if e.get("event") == "OM_EXTRACTOR_PATH"]
    assert path_events, "OM_EXTRACTOR_PATH must be emitted"
    evt = path_events[0]
    assert evt["extractor_mode"] == "model"
    assert evt["edges"] >= 1, "edges field in event must reflect edge count"
    assert evt["nodes"] >= 1


def test_model_path_strict_mode_field_true_in_event(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture
) -> None:
    """OM_EXTRACTOR_PATH event includes strict_mode=True when in strict mode."""
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test-key")
    monkeypatch.delenv("OM_EXTRACTOR_STRICT", raising=False)
    monkeypatch.delenv("OM_EXTRACTOR_MODE", raising=False)

    with patch("scripts.om_compressor._call_llm_extract", return_value=_fake_chunk_with_edges()):
        om_compressor._extract_items(_make_messages(), _make_cfg())

    captured = capsys.readouterr()
    events = [json.loads(ln) for ln in captured.out.splitlines() if ln.strip() and '"event"' in ln]
    path_events = [e for e in events if e.get("event") == "OM_EXTRACTOR_PATH"]
    assert path_events[0].get("strict_mode") is True


# ---------------------------------------------------------------------------
# OM-2: _call_llm_extract — parsing and allowlist enforcement (mock HTTP)
# ---------------------------------------------------------------------------


def _make_fake_http_response(payload: dict) -> MagicMock:
    """Build a minimal fake urllib response returning the given JSON payload."""

    body = json.dumps(payload).encode("utf-8")
    mock_resp = MagicMock()
    mock_resp.read.return_value = body
    mock_resp.__enter__ = lambda s: s
    mock_resp.__exit__ = MagicMock(return_value=False)
    return mock_resp


def _openai_response_wrapping(content_str: str) -> dict:
    """Wrap a content string in a minimal OpenAI chat completion response shape."""
    return {
        "choices": [{"message": {"role": "assistant", "content": content_str}}]
    }


def test_call_llm_extract_parses_nodes_and_edges(monkeypatch: pytest.MonkeyPatch) -> None:
    """_call_llm_extract correctly parses nodes and edges from a valid LLM response."""
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test-key")

    msgs = _make_messages(2)
    # Use a chat-compatible model so the mock chat-completions response is parsed correctly.
    cfg = _make_cfg(model_id="gpt-4o")

    llm_payload = {
        "nodes": [
            {
                "node_type": "Preference",
                "semantic_domain": "sessions_main",
                "content": "Prefers bullet-point summaries",
                "urgency_score": 3,
                "source_message_ids": [msgs[0].message_id],
            },
            {
                "node_type": "OperationalRule",
                "semantic_domain": "sessions_main",
                "content": "Always add Google Meet link to calendar events",
                "urgency_score": 4,
                "source_message_ids": [msgs[1].message_id],
            },
        ],
        "edges": [
            {"source_index": 0, "target_index": 1, "relation_type": "MOTIVATES"},
        ],
    }
    fake_resp = _make_fake_http_response(_openai_response_wrapping(json.dumps(llm_payload)))

    with patch("urllib.request.urlopen", return_value=fake_resp):
        result = om_compressor._call_llm_extract(msgs, cfg)

    assert len(result.nodes) == 2
    assert len(result.edges) == 1
    assert result.edges[0].relation_type == "MOTIVATES"
    assert result.nodes[0].node_type == "Preference"
    assert result.nodes[1].node_type == "OperationalRule"


def test_call_llm_extract_drops_illegal_relation_type(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture
) -> None:
    """_call_llm_extract silently drops edges with illegal relation types."""
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test-key")

    msgs = _make_messages(2)
    cfg = _make_cfg(model_id="gpt-4o")  # chat-compatible model → chat API mock response

    llm_payload = {
        "nodes": [
            {"node_type": "WorldState", "semantic_domain": "sessions_main",
             "content": "Node A", "urgency_score": 3, "source_message_ids": []},
            {"node_type": "WorldState", "semantic_domain": "sessions_main",
             "content": "Node B", "urgency_score": 3, "source_message_ids": []},
        ],
        "edges": [
            # "RELATES_TO" is not in RELATION_TYPES allowlist
            {"source_index": 0, "target_index": 1, "relation_type": "RELATES_TO"},
            # "MOTIVATES" is valid
            {"source_index": 0, "target_index": 1, "relation_type": "MOTIVATES"},
        ],
    }
    fake_resp = _make_fake_http_response(_openai_response_wrapping(json.dumps(llm_payload)))

    with patch("urllib.request.urlopen", return_value=fake_resp):
        result = om_compressor._call_llm_extract(msgs, cfg)

    # RELATES_TO dropped, MOTIVATES kept
    assert len(result.edges) == 1
    assert result.edges[0].relation_type == "MOTIVATES"

    captured = capsys.readouterr()
    events = [json.loads(ln) for ln in captured.out.splitlines() if ln.strip() and '"event"' in ln]
    blocked = [e for e in events if e.get("event") == "OM_RELATION_TYPE_INTERPOLATION_BLOCKED"]
    assert len(blocked) >= 1
    assert blocked[0]["relation_type"] == "RELATES_TO"


def test_call_llm_extract_raises_on_no_api_key(monkeypatch: pytest.MonkeyPatch) -> None:
    """_call_llm_extract raises OMCompressorError when no API key is set."""
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("OM_EXTRACTOR_API_KEY", raising=False)

    with pytest.raises(om_compressor.OMCompressorError, match="no API key"):
        om_compressor._call_llm_extract(_make_messages(), _make_cfg(model_id="gpt-4o"))


def test_call_llm_extract_raises_on_invalid_json(monkeypatch: pytest.MonkeyPatch) -> None:
    """_call_llm_extract raises OMCompressorError when LLM returns non-JSON content."""
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test-key")

    bad_resp = _make_fake_http_response(
        _openai_response_wrapping("This is plain text, not JSON")
    )
    with (
        patch("urllib.request.urlopen", return_value=bad_resp),
        pytest.raises(om_compressor.OMCompressorError, match="not valid JSON"),
    ):
        # Use chat-compatible model so the mock chat response envelope is parsed correctly.
        om_compressor._call_llm_extract(_make_messages(), _make_cfg(model_id="gpt-4o"))


# ---------------------------------------------------------------------------
# NEW: BLOCKER #1 — responses API auto-style selection + fail-fast validation
# ---------------------------------------------------------------------------


def test_resolve_llm_api_style_auto_selects_responses_for_codex(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """_resolve_llm_api_style auto-selects 'responses' for gpt-5.1-codex-mini."""
    monkeypatch.delenv("OM_COMPRESSOR_LLM_API_STYLE", raising=False)
    assert om_compressor._resolve_llm_api_style("gpt-5.1-codex-mini") == "responses"


def test_resolve_llm_api_style_auto_selects_responses_for_o_series(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """_resolve_llm_api_style auto-selects 'responses' for o1, o3, o4-mini etc."""
    monkeypatch.delenv("OM_COMPRESSOR_LLM_API_STYLE", raising=False)
    for model in ("o1", "o3", "o4-mini", "o1-mini", "openai/o3", "openrouter/openai/o4-mini"):
        assert om_compressor._resolve_llm_api_style(model) == "responses", (
            f"Expected 'responses' for {model!r}"
        )


def test_resolve_llm_api_style_auto_selects_chat_for_gpt4o(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """_resolve_llm_api_style defaults to 'chat' for non-codex/o-series models."""
    monkeypatch.delenv("OM_COMPRESSOR_LLM_API_STYLE", raising=False)
    for model in ("gpt-4o", "gpt-4.1", "claude-3-opus", "anthropic/claude-3-opus"):
        assert om_compressor._resolve_llm_api_style(model) == "chat", (
            f"Expected 'chat' for {model!r}"
        )


def test_resolve_llm_api_style_explicit_env_honored(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Explicit OM_COMPRESSOR_LLM_API_STYLE=responses is honoured for any model."""
    monkeypatch.setenv("OM_COMPRESSOR_LLM_API_STYLE", "responses")
    assert om_compressor._resolve_llm_api_style("gpt-4o") == "responses"


def test_resolve_llm_api_style_explicit_chat_with_codex_raises(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Explicit 'chat' style + codex/o-series model → OMCompressorError (fail-fast)."""
    monkeypatch.setenv("OM_COMPRESSOR_LLM_API_STYLE", "chat")
    with pytest.raises(om_compressor.OMCompressorError, match="requires the 'responses' API"):
        om_compressor._resolve_llm_api_style("gpt-5.1-codex-mini")


def test_resolve_llm_api_style_explicit_chat_with_o4_raises(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Explicit 'chat' style + o4-mini → OMCompressorError (fail-fast)."""
    monkeypatch.setenv("OM_COMPRESSOR_LLM_API_STYLE", "chat")
    with pytest.raises(om_compressor.OMCompressorError, match="requires the 'responses' API"):
        om_compressor._resolve_llm_api_style("o4-mini")


def test_resolve_llm_api_style_no_model_defaults_chat(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """With no model_id and no env var, default style is 'chat'."""
    monkeypatch.delenv("OM_COMPRESSOR_LLM_API_STYLE", raising=False)
    assert om_compressor._resolve_llm_api_style(None) == "chat"
    assert om_compressor._resolve_llm_api_style("") == "chat"


# ---------------------------------------------------------------------------
# NEW: BLOCKER #1 — responses API response path parsing in _call_llm_extract
# ---------------------------------------------------------------------------


def _responses_api_response_wrapping(content_str: str) -> dict:
    """Wrap a content string in a minimal /v1/responses API response shape."""
    return {
        "output": [
            {
                "type": "message",
                "content": [
                    {"type": "output_text", "text": content_str},
                ],
            }
        ]
    }


def test_call_llm_extract_responses_api_path_parsed(monkeypatch: pytest.MonkeyPatch) -> None:
    """_call_llm_extract correctly parses a /v1/responses format response for codex models."""
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test-key")
    monkeypatch.delenv("OM_COMPRESSOR_LLM_API_STYLE", raising=False)

    msgs = _make_messages(2)
    cfg = _make_cfg(model_id="gpt-5.1-codex-mini")  # auto-detects → responses style

    llm_payload = {
        "nodes": [
            {
                "node_type": "OperationalRule",
                "semantic_domain": "sessions_main",
                "content": "Always use responses API for codex models",
                "urgency_score": 4,
                "source_message_ids": [msgs[0].message_id],
            },
        ],
        "edges": [],
    }
    fake_resp = _make_fake_http_response(
        _responses_api_response_wrapping(json.dumps(llm_payload))
    )

    with patch("urllib.request.urlopen", return_value=fake_resp):
        result = om_compressor._call_llm_extract(msgs, cfg)

    assert len(result.nodes) == 1
    assert result.nodes[0].node_type == "OperationalRule"
    assert len(result.edges) == 0


def test_call_llm_extract_responses_api_empty_output_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    """_call_llm_extract raises on responses API shape with no output_text."""
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test-key")
    monkeypatch.delenv("OM_COMPRESSOR_LLM_API_STYLE", raising=False)

    msgs = _make_messages(1)
    cfg = _make_cfg(model_id="o4-mini")  # auto-detects → responses style

    # Return a /v1/responses response with no output_text content item
    bad_payload = {"output": [{"type": "function_call", "content": []}]}
    fake_resp = _make_fake_http_response(bad_payload)

    with (
        patch("urllib.request.urlopen", return_value=fake_resp),
        pytest.raises(om_compressor.OMCompressorError, match="no output_text"),
    ):
        om_compressor._call_llm_extract(msgs, cfg)


# ---------------------------------------------------------------------------
# NEW: HIGH — duplicate-key detection in JSON parsing
# ---------------------------------------------------------------------------


def test_detect_duplicate_keys_raises_on_duplicate(monkeypatch: pytest.MonkeyPatch) -> None:
    """_detect_duplicate_keys raises OMCompressorError on a duplicate JSON key."""
    dup_json = '{"a": 1, "b": 2, "a": 3}'  # "a" appears twice
    with pytest.raises(om_compressor.OMCompressorError, match="Duplicate JSON key"):
        import json as _json
        _json.loads(dup_json, object_pairs_hook=om_compressor._detect_duplicate_keys)


def test_detect_duplicate_keys_allows_unique_keys(monkeypatch: pytest.MonkeyPatch) -> None:
    """_detect_duplicate_keys passes through JSON with unique keys unchanged."""
    import json as _json
    result = _json.loads('{"a": 1, "b": 2}', object_pairs_hook=om_compressor._detect_duplicate_keys)
    assert result == {"a": 1, "b": 2}


def test_call_llm_extract_rejects_duplicate_key_in_envelope(monkeypatch: pytest.MonkeyPatch) -> None:
    """_call_llm_extract raises OMCompressorError when the LLM API envelope has duplicate keys."""
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test-key")

    msgs = _make_messages(1)
    cfg = _make_cfg(model_id="gpt-4o")

    # HTTP response body has duplicate "id" key in the envelope JSON
    dup_envelope = b'{"id": "1", "id": "2", "choices": [{"message": {"content": "{}"}}]}'
    mock_resp = MagicMock()
    mock_resp.read.return_value = dup_envelope
    mock_resp.__enter__ = lambda s: s
    mock_resp.__exit__ = MagicMock(return_value=False)

    with (
        patch("urllib.request.urlopen", return_value=mock_resp),
        pytest.raises(om_compressor.OMCompressorError, match="Duplicate JSON key"),
    ):
        om_compressor._call_llm_extract(msgs, cfg)


def test_call_llm_extract_rejects_duplicate_key_in_payload(monkeypatch: pytest.MonkeyPatch) -> None:
    """_call_llm_extract raises OMCompressorError when the extracted JSON payload has duplicate keys."""
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test-key")

    msgs = _make_messages(1)
    cfg = _make_cfg(model_id="gpt-4o")

    # Model output JSON has duplicate "nodes" key
    dup_payload_str = '{"nodes": [], "nodes": [], "edges": []}'
    fake_resp = _make_fake_http_response(_openai_response_wrapping(dup_payload_str))

    with (
        patch("urllib.request.urlopen", return_value=fake_resp),
        pytest.raises(om_compressor.OMCompressorError, match="Duplicate JSON key"),
    ):
        om_compressor._call_llm_extract(msgs, cfg)


# ---------------------------------------------------------------------------
# NEW: SSRF hardening — _is_ssrf_blocked_host and _llm_chat_base_url
# ---------------------------------------------------------------------------


def test_is_ssrf_blocked_host_blocks_localhost(monkeypatch: pytest.MonkeyPatch) -> None:
    assert om_compressor._is_ssrf_blocked_host("localhost") is True
    assert om_compressor._is_ssrf_blocked_host("localhost:8080") is True


def test_is_ssrf_blocked_host_blocks_loopback_ip(monkeypatch: pytest.MonkeyPatch) -> None:
    assert om_compressor._is_ssrf_blocked_host("127.0.0.1") is True
    assert om_compressor._is_ssrf_blocked_host("127.0.0.1:11434") is True


def test_is_ssrf_blocked_host_blocks_private_ip(monkeypatch: pytest.MonkeyPatch) -> None:
    assert om_compressor._is_ssrf_blocked_host("10.0.0.1") is True
    assert om_compressor._is_ssrf_blocked_host("192.168.1.1") is True
    assert om_compressor._is_ssrf_blocked_host("172.16.0.1") is True


def test_is_ssrf_blocked_host_blocks_link_local(monkeypatch: pytest.MonkeyPatch) -> None:
    assert om_compressor._is_ssrf_blocked_host("169.254.169.254") is True


def test_is_ssrf_blocked_host_allows_external_hostname(monkeypatch: pytest.MonkeyPatch) -> None:
    assert om_compressor._is_ssrf_blocked_host("api.openai.com") is False
    assert om_compressor._is_ssrf_blocked_host("openrouter.ai") is False


def test_llm_chat_base_url_blocks_localhost_by_default(monkeypatch: pytest.MonkeyPatch) -> None:
    """_llm_chat_base_url raises for local LLM URLs when OM_ALLOW_LOCAL_LLM is unset."""
    monkeypatch.setenv("OM_COMPRESSOR_LLM_BASE_URL", "http://localhost:11434/v1")
    monkeypatch.delenv("OM_ALLOW_LOCAL_LLM", raising=False)
    with pytest.raises(om_compressor.OMCompressorError, match="private/loopback"):
        om_compressor._llm_chat_base_url()


def test_llm_chat_base_url_allows_localhost_with_bypass(monkeypatch: pytest.MonkeyPatch) -> None:
    """OM_ALLOW_LOCAL_LLM=1 bypasses the SSRF block for local LLM endpoints."""
    monkeypatch.setenv("OM_COMPRESSOR_LLM_BASE_URL", "http://localhost:11434/v1")
    monkeypatch.setenv("OM_ALLOW_LOCAL_LLM", "1")
    url = om_compressor._llm_chat_base_url()
    assert url == "http://localhost:11434/v1"


def test_llm_chat_base_url_allows_external_provider(monkeypatch: pytest.MonkeyPatch) -> None:
    """_llm_chat_base_url accepts standard external provider URLs without bypass."""
    monkeypatch.setenv("OM_COMPRESSOR_LLM_BASE_URL", "https://openrouter.ai/api/v1")
    monkeypatch.delenv("OM_ALLOW_LOCAL_LLM", raising=False)
    url = om_compressor._llm_chat_base_url()
    assert url == "https://openrouter.ai/api/v1"


# ---------------------------------------------------------------------------
# FM1: graphiti-context and graphiti-fallback XML wrapper stripping
# ---------------------------------------------------------------------------


def test_strip_graphiti_context_wrapper_removed() -> None:
    """<graphiti-context>…</graphiti-context> block is fully stripped."""
    content = (
        "Pre-context message text.\n\n"
        "<graphiti-context>\n"
        "## Graphiti Recall\n"
        "- Some injected context line 1\n"
        "- Some injected context line 2\n"
        "</graphiti-context>\n\n"
        "Post-context message text."
    )
    result = om_compressor.strip_untrusted_metadata(content)
    assert "<graphiti-context>" not in result
    assert "</graphiti-context>" not in result
    assert "Graphiti Recall" not in result
    assert "injected context" not in result
    assert "Pre-context message text." in result
    assert "Post-context message text." in result


def test_strip_graphiti_fallback_wrapper_removed() -> None:
    """<graphiti-fallback>…</graphiti-fallback> block is fully stripped."""
    content = (
        "Before fallback.\n"
        "<graphiti-fallback>\n"
        "Fallback memory content injected here.\n"
        "More fallback lines.\n"
        "</graphiti-fallback>\n"
        "After fallback."
    )
    result = om_compressor.strip_untrusted_metadata(content)
    assert "<graphiti-fallback>" not in result
    assert "</graphiti-fallback>" not in result
    assert "Fallback memory content" not in result
    assert "Before fallback." in result
    assert "After fallback." in result


def test_strip_graphiti_context_nested_content_fully_removed() -> None:
    """Nested multi-line content inside <graphiti-context> is entirely stripped."""
    inner_lines = "\n".join(f"- nested line {i}" for i in range(50))
    content = (
        "Header.\n"
        f"<graphiti-context>\n"
        f"{inner_lines}\n"
        "</graphiti-context>\n"
        "Footer."
    )
    result = om_compressor.strip_untrusted_metadata(content)
    assert "<graphiti-context>" not in result
    assert "nested line" not in result
    assert "Header." in result
    assert "Footer." in result


def test_strip_graphiti_both_wrappers_in_one_message() -> None:
    """Both graphiti wrapper types in one message are both stripped."""
    content = (
        "Start.\n"
        "<graphiti-context>\n"
        "Context block content.\n"
        "</graphiti-context>\n"
        "Middle.\n"
        "<graphiti-fallback>\n"
        "Fallback block content.\n"
        "</graphiti-fallback>\n"
        "End."
    )
    result = om_compressor.strip_untrusted_metadata(content)
    assert "Context block content." not in result
    assert "Fallback block content." not in result
    assert "Start." in result
    assert "Middle." in result
    assert "End." in result


def test_strip_graphiti_context_unclosed_tag_is_preserved() -> None:
    """An unclosed <graphiti-context> tag (no closing tag within 1000 lines) is NOT stripped."""
    content = (
        "<graphiti-context>\n"
        "Orphaned content with no closing tag.\n"
    )
    # Without a closing tag, the bounded scanner gives up and leaves the block intact.
    result = om_compressor.strip_untrusted_metadata(content)
    assert "<graphiti-context>" in result
    assert "Orphaned content" in result


def test_strip_graphiti_wrappers_exported_from_module() -> None:
    """_UNTRUSTED_XML_WRAPPERS dict is accessible and contains both graphiti keys."""
    wrappers = om_compressor._UNTRUSTED_XML_WRAPPERS
    assert "<graphiti-context>" in wrappers
    assert "<graphiti-fallback>" in wrappers
    assert wrappers["<graphiti-context>"] == "</graphiti-context>"
    assert wrappers["<graphiti-fallback>"] == "</graphiti-fallback>"


# ---------------------------------------------------------------------------
# FM3: SUPERSEDES prompt guidance present + chronological ordering contract
# ---------------------------------------------------------------------------


def test_om_extract_system_prompt_contains_supersedes_guidance() -> None:
    """_OM_EXTRACT_SYSTEM_PROMPT instructs model to use SUPERSEDES for temporal updates."""
    prompt = om_compressor._OM_EXTRACT_SYSTEM_PROMPT
    # Must mention SUPERSEDES in the temporal sequencing context
    assert "SUPERSEDES" in prompt, "SUPERSEDES must appear in extraction system prompt"
    # Must explain the semantic: newer → older direction
    assert "newer" in prompt.lower() or "chronolog" in prompt.lower(), (
        "Prompt must explain chronological/newer-to-older SUPERSEDES semantics"
    )
    # Must mention chronological ordering of messages
    assert "chronolog" in prompt.lower() or "oldest first" in prompt.lower(), (
        "Prompt must state messages are in chronological order"
    )


def test_fetch_messages_by_ids_returns_chronological_order() -> None:
    """_fetch_messages_by_ids returns rows in created_at ASC order (DB fetch order), not
    input message_ids order — ensuring temporal sequencing for the extractor."""
    # Simulate rows returned by the DB already sorted by created_at ASC
    # (msg3 earliest, msg1 latest — opposite of the input order)
    fake_rows = [
        {
            "message_id": "msg3",
            "source_session_id": "sess1",
            "content": "earliest message",
            "created_at": "2026-01-01T10:00:00Z",
            "content_embedding": [],
            "om_extract_attempts": 0,
        },
        {
            "message_id": "msg1",
            "source_session_id": "sess1",
            "content": "middle message",
            "created_at": "2026-01-01T11:00:00Z",
            "content_embedding": [],
            "om_extract_attempts": 0,
        },
        {
            "message_id": "msg2",
            "source_session_id": "sess1",
            "content": "latest message",
            "created_at": "2026-01-01T12:00:00Z",
            "content_embedding": [],
            "om_extract_attempts": 0,
        },
    ]

    # Mock session.run().data() to return chronologically sorted rows
    mock_result = MagicMock()
    mock_result.data.return_value = fake_rows
    mock_session = MagicMock()
    mock_session.run.return_value = mock_result

    # Input order is deliberately reverse-chronological (msg2, msg1, msg3)
    # The function must return in DB row order (msg3, msg1, msg2 = created_at ASC)
    result = om_compressor._fetch_messages_by_ids(mock_session, ["msg2", "msg1", "msg3"])

    assert len(result) == 3
    assert result[0].message_id == "msg3", (
        f"First message should be earliest (msg3), got {result[0].message_id}"
    )
    assert result[1].message_id == "msg1", (
        f"Second message should be msg1, got {result[1].message_id}"
    )
    assert result[2].message_id == "msg2", (
        f"Third message should be latest (msg2), got {result[2].message_id}"
    )


# ---------------------------------------------------------------------------
# NO-GO fixes: IPv6 parsing, OM_ALLOW_LOCAL_LLM scope, cloud-metadata always-blocked
# ---------------------------------------------------------------------------


def test_is_ssrf_blocked_host_blocks_ipv6_loopback() -> None:
    """[::1] IPv6 loopback must be correctly parsed and blocked.

    The naive netloc.split(':')[0] approach returns '[', which fails ip_address()
    and causes the IPv6 loopback to be silently allowed.  urlparse.hostname fixes this.
    """
    assert om_compressor._is_ssrf_blocked_host("[::1]") is True
    assert om_compressor._is_ssrf_blocked_host("[::1]:8080") is True
    assert om_compressor._is_ssrf_blocked_host("[::1]:11434") is True


def test_is_ssrf_blocked_host_blocks_ipv6_link_local() -> None:
    """IPv6 link-local addresses (fe80::) must be blocked."""
    assert om_compressor._is_ssrf_blocked_host("[fe80::1]") is True
    assert om_compressor._is_ssrf_blocked_host("[fe80::1]:8080") is True


def test_is_ssrf_blocked_host_blocks_ipv6_private() -> None:
    """IPv6 private / ULA addresses (fc00::/7) must be blocked."""
    assert om_compressor._is_ssrf_blocked_host("[fd00::1]") is True


def test_llm_chat_base_url_blocks_ipv6_loopback_by_default(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """IPv6 loopback [::1] is blocked by default (no OM_ALLOW_LOCAL_LLM)."""
    monkeypatch.setenv("OM_COMPRESSOR_LLM_BASE_URL", "http://[::1]:11434/v1")
    monkeypatch.delenv("OM_ALLOW_LOCAL_LLM", raising=False)
    with pytest.raises(om_compressor.OMCompressorError, match="private/loopback"):
        om_compressor._llm_chat_base_url()


def test_llm_chat_base_url_allows_ipv6_loopback_with_bypass(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """OM_ALLOW_LOCAL_LLM=1 should permit [::1] for local Ollama/dev usage."""
    monkeypatch.setenv("OM_COMPRESSOR_LLM_BASE_URL", "http://[::1]:11434/v1")
    monkeypatch.setenv("OM_ALLOW_LOCAL_LLM", "1")
    url = om_compressor._llm_chat_base_url()
    assert url == "http://[::1]:11434/v1"


def test_llm_chat_base_url_always_blocks_cloud_metadata_even_with_bypass(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Cloud metadata 169.254.169.254 is ALWAYS blocked, even with OM_ALLOW_LOCAL_LLM=1.

    This is the key NO-GO fix: OM_ALLOW_LOCAL_LLM must only widen the allowed set
    to localhost/127.x/[::1], never to link-local/cloud-metadata endpoints.
    """
    monkeypatch.setenv("OM_COMPRESSOR_LLM_BASE_URL", "http://169.254.169.254/latest/meta-data/")
    monkeypatch.setenv("OM_ALLOW_LOCAL_LLM", "1")
    with pytest.raises(om_compressor.OMCompressorError, match="link-local"):
        om_compressor._llm_chat_base_url()


def test_llm_chat_base_url_always_blocks_ipv6_link_local_even_with_bypass(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """IPv6 link-local fe80::1 is ALWAYS blocked, even with OM_ALLOW_LOCAL_LLM=1."""
    monkeypatch.setenv("OM_COMPRESSOR_LLM_BASE_URL", "http://[fe80::1]/v1")
    monkeypatch.setenv("OM_ALLOW_LOCAL_LLM", "1")
    with pytest.raises(om_compressor.OMCompressorError, match="link-local"):
        om_compressor._llm_chat_base_url()


def test_llm_chat_base_url_allows_127_0_0_1_with_bypass(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """OM_ALLOW_LOCAL_LLM=1 allows 127.0.0.1 (IPv4 loopback) for local dev."""
    monkeypatch.setenv("OM_COMPRESSOR_LLM_BASE_URL", "http://127.0.0.1:11434/v1")
    monkeypatch.setenv("OM_ALLOW_LOCAL_LLM", "1")
    url = om_compressor._llm_chat_base_url()
    assert url == "http://127.0.0.1:11434/v1"
