from dataclasses import dataclass, field
from enum import Enum
from types import SimpleNamespace
from datetime import datetime, timezone
from unittest.mock import MagicMock

import pytest

# Import function under test directly
import importlib.util
import pathlib

BASE_DIR = pathlib.Path(__file__).resolve().parents[1]

spec = importlib.util.spec_from_file_location(
    "bulk_utils", BASE_DIR.parent / "graphiti_core" / "utils" / "bulk_utils.py"
)
bulk_utils = importlib.util.module_from_spec(spec)
spec.loader.exec_module(bulk_utils)

add_nodes_and_edges_bulk_tx = bulk_utils.add_nodes_and_edges_bulk_tx


class EpisodeType(Enum):
    message = "message"


@dataclass
class DummyEpisode:
    name: str
    group_id: str
    labels: list[str] = field(default_factory=list)
    source: EpisodeType = EpisodeType.message
    source_description: str = "desc"
    content: str = "content"
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    valid_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    def model_dump(self):
        return {
            "name": self.name,
            "group_id": self.group_id,
            "labels": self.labels,
            "source": self.source,
            "source_description": self.source_description,
            "content": self.content,
            "created_at": self.created_at,
            "valid_at": self.valid_at,
            "uuid": "u",
        }


class DummyTx:
    def __init__(self):
        self.calls = []

    async def run(self, query, **kwargs):
        self.calls.append((query, kwargs))
        return None


@pytest.mark.asyncio
async def test_add_nodes_and_edges_bulk_tx_converts_datetimes():
    now = datetime.now(timezone.utc)
    episode = DummyEpisode(name="ep", group_id="g", created_at=now, valid_at=now)

    tx = DummyTx()

    await add_nodes_and_edges_bulk_tx(
        tx,
        episodic_nodes=[episode],
        episodic_edges=[],
        entity_nodes=[],
        entity_edges=[],
        embedder=MagicMock(),
        driver=SimpleNamespace(provider="falkordb"),
    )

    assert tx.calls
    _, params = tx.calls[0]
    ep = params["episodes"][0]
    assert ep["created_at"] == now.isoformat()
    assert ep["valid_at"] == now.isoformat()
    assert ep["source"] == "message"
