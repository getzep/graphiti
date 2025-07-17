from datetime import datetime, timezone

from graphiti_core.nodes import EpisodeType, EpisodicNode
from graphiti_core.utils.bulk_utils import RawEpisode


def test_raw_episode_metadata_preserved():
    meta = {"foo": "bar"}
    raw = RawEpisode(
        name="ep1",
        content="content",
        source_description="src",
        source=EpisodeType.text,
        reference_time=datetime.now(timezone.utc),
        metadata=meta,
    )
    assert raw.metadata == meta

    node = EpisodicNode(
        name=raw.name,
        group_id="g",
        labels=[],
        source=raw.source,
        content=raw.content,
        source_description=raw.source_description,
        created_at=raw.reference_time,
        valid_at=raw.reference_time,
        metadata=raw.metadata,
    )
    assert node.metadata == meta
