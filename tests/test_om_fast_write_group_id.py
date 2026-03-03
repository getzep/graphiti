import os
import unittest
from unittest.mock import patch

from scripts import om_fast_write


class TestOmFastWriteGroupId(unittest.TestCase):
    def test_default_group_id_when_missing(self) -> None:
        raw = {
            "source_session_id": "agent:main:telegram:group:-100:test",
            "role": "assistant",
            "content": "hello",
            "created_at": "2026-03-03T20:00:00Z",
        }
        with patch.dict(os.environ, {}, clear=False):
            os.environ.pop("OM_GROUP_ID", None)
            payload = om_fast_write._coerce_payload(raw)
        self.assertEqual(payload.group_id, "s1_observational_memory")

    def test_explicit_group_id_wins(self) -> None:
        raw = {
            "source_session_id": "agent:main:telegram:group:-100:test",
            "role": "assistant",
            "content": "hello",
            "created_at": "2026-03-03T20:00:00Z",
            "group_id": "s1_observational_memory",
        }
        payload = om_fast_write._coerce_payload(raw)
        self.assertEqual(payload.group_id, "s1_observational_memory")

    def test_rejects_invalid_group_id(self) -> None:
        raw = {
            "source_session_id": "agent:main:telegram:group:-100:test",
            "role": "assistant",
            "content": "hello",
            "created_at": "2026-03-03T20:00:00Z",
            "group_id": "bad lane with spaces",
        }
        with self.assertRaises(RuntimeError):
            om_fast_write._coerce_payload(raw)


if __name__ == "__main__":
    unittest.main()
