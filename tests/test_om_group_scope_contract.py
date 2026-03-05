from pathlib import Path


def test_om_compressor_scopes_message_backlog_to_om_group_id() -> None:
    src = Path("scripts/om_compressor.py").read_text(encoding="utf-8")
    assert "coalesce(m.group_id, $group_id) = $group_id" in src
    assert "def om_group_id(" in src


def test_om_fast_write_sets_group_id_on_episode_and_message() -> None:
    src = Path("scripts/om_fast_write.py").read_text(encoding="utf-8")
    assert "e.group_id = $group_id" in src
    assert "m.group_id = $group_id" in src
    assert "DEFAULT_OM_GROUP_ID = \"s1_observational_memory\"" in src


def test_om_compressor_index_maintenance_is_non_destructive() -> None:
    src = Path("scripts/om_compressor.py").read_text(encoding="utf-8")
    assert "DROP INDEX omnode_content_fulltext IF EXISTS" not in src
    assert (
        "CREATE FULLTEXT INDEX omnode_content_fulltext IF NOT EXISTS FOR (n:OMNode) "
        "ON EACH [n.content, n.group_id]" in src
    )
