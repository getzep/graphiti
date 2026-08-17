from datetime import datetime, timedelta, timezone
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from graph_service.config import Settings
from graph_service.sources.connectors import SourceConnector
from graph_service.sources.models import SourceDocument
from graph_service.sources.store import SourceStore
from graph_service.sources.sync import SyncManager


class StubConnector(SourceConnector):
    def __init__(
        self,
        documents: list[SourceDocument],
        *,
        seen: set[str] | None = None,
        errors: list[str] | None = None,
        inventory_complete: bool = True,
    ):
        super().__init__()
        self.documents = documents
        self.seen_external_ids = seen or {document.external_id for document in documents}
        self.errors = errors or []
        self.inventory_complete = inventory_complete

    async def fetch(
        self, *, watermark_ms: int | None = None, full_sync: bool = False
    ) -> list[SourceDocument]:
        del watermark_ms, full_sync
        return self.documents


class StubGraphiti:
    def __init__(self, episode_uuid: str = 'generated-episode'):
        self.add_episode = AsyncMock(
            return_value=SimpleNamespace(episode=SimpleNamespace(uuid=episode_uuid))
        )
        self.close = AsyncMock()


def _settings(tmp_path) -> Settings:
    return Settings(
        _env_file=None,
        source_state_path=tmp_path / 'state.db',
        upload_root=tmp_path / 'uploads',
        openai_api_key='offline-test-key',
        model_name='offline-test-model',
        neo4j_uri='bolt://unused',
        neo4j_user='unused',
        neo4j_password='unused',
    )


def _document(
    external_id: str,
    content: str,
    *,
    version: str,
    title: str,
    updated_at: datetime,
) -> SourceDocument:
    return SourceDocument(
        external_id=external_id,
        title=title,
        content=content,
        updated_at=updated_at,
        remote_version=version,
    )


async def _execute(manager: SyncManager, store: SourceStore, source_id: str) -> dict:
    job = store.create_job(source_id, full_sync=False)
    await manager._execute(job, store.get_source(source_id))
    return store.get_job(job['id'])


@pytest.mark.asyncio
async def test_sync_is_incremental_and_refreshes_metadata_for_unchanged_content(
    tmp_path, monkeypatch
):
    settings = _settings(tmp_path)
    store = SourceStore(settings.source_state_path)
    manager = SyncManager(settings, store)
    source = store.create_source(
        kind='local', name='Local', group_id='neo4j', config={}, enabled=True
    )
    graphiti = StubGraphiti()

    first_time = datetime(2026, 1, 1, tzinfo=timezone.utc)
    connectors = [
        StubConnector(
            [_document('guide.md', 'same body', version='1', title='旧标题', updated_at=first_time)]
        ),
        StubConnector(
            [
                _document(
                    'guide.md',
                    'same body',
                    version='2',
                    title='新标题',
                    updated_at=first_time + timedelta(days=1),
                )
            ]
        ),
    ]
    monkeypatch.setattr(
        'graph_service.sources.sync.build_connector', lambda *_args, **_kwargs: connectors.pop(0)
    )
    monkeypatch.setattr(
        'graph_service.sources.sync.create_graphiti_client', lambda _settings: graphiti
    )

    first_job = await _execute(manager, store, source['id'])
    second_job = await _execute(manager, store, source['id'])

    assert first_job['status'] == 'succeeded'
    assert first_job['created'] == 1
    assert second_job['status'] == 'succeeded'
    assert second_job['skipped'] == 1
    assert graphiti.add_episode.await_count == 1
    assert 'uuid' not in graphiti.add_episode.await_args.kwargs
    item = store.get_item(source['id'], 'guide.md')
    assert item is not None
    assert item['episode_uuid'] == 'generated-episode'
    assert item['remote_version'] == '2'
    assert item['title'] == '新标题'
    assert item['source_updated_at'] == '2026-01-02T00:00:00+00:00'


@pytest.mark.asyncio
async def test_partial_feishu_fetch_does_not_tombstone_seen_failed_document(tmp_path, monkeypatch):
    settings = _settings(tmp_path)
    store = SourceStore(settings.source_state_path)
    manager = SyncManager(settings, store)
    source = store.create_source(
        kind='feishu',
        name='Feishu',
        group_id='neo4j',
        config={'folder_token': 'folder'},
        enabled=True,
    )
    updated_at = datetime(2026, 1, 1, tzinfo=timezone.utc)
    for external_id in ('doc-a', 'doc-b'):
        document = _document(
            external_id, 'body', version='1', title=external_id, updated_at=updated_at
        )
        store.upsert_item(
            source_id=source['id'],
            external_id=external_id,
            remote_version='1',
            content_hash=document.content_hash,
            episode_uuid=f'episode-{external_id}',
            title=external_id,
            source_updated_at=updated_at.isoformat(),
        )

    connector = StubConnector(
        [_document('doc-a', 'body', version='1', title='doc-a', updated_at=updated_at)],
        seen={'doc-a', 'doc-b'},
        errors=['doc-b: transient download failure'],
    )
    monkeypatch.setattr(
        'graph_service.sources.sync.build_connector', lambda *_args, **_kwargs: connector
    )

    job = await _execute(manager, store, source['id'])

    assert job['status'] == 'partial'
    assert job['failed'] == 1
    assert store.get_item(source['id'], 'doc-a')['deleted_at'] is None
    assert store.get_item(source['id'], 'doc-b')['deleted_at'] is None


@pytest.mark.asyncio
async def test_changed_document_uses_previous_episode_and_persists_generated_uuid(
    tmp_path, monkeypatch
):
    settings = _settings(tmp_path)
    store = SourceStore(settings.source_state_path)
    manager = SyncManager(settings, store)
    source = store.create_source(
        kind='local', name='Local', group_id='neo4j', config={}, enabled=True
    )
    updated_at = datetime(2026, 1, 1, tzinfo=timezone.utc)
    old_document = _document(
        'guide.pdf', 'old body', version='1', title='guide.pdf', updated_at=updated_at
    )
    store.upsert_item(
        source_id=source['id'],
        external_id=old_document.external_id,
        remote_version=old_document.remote_version,
        content_hash=old_document.content_hash,
        episode_uuid='previous-episode',
        title=old_document.title,
        source_updated_at=old_document.updated_at.isoformat(),
    )
    new_document = _document(
        'guide.pdf',
        'new body',
        version='2',
        title='guide.pdf',
        updated_at=updated_at + timedelta(days=1),
    )
    connector = StubConnector([new_document])
    graphiti = StubGraphiti('new-generated-episode')
    monkeypatch.setattr(
        'graph_service.sources.sync.build_connector', lambda *_args, **_kwargs: connector
    )
    monkeypatch.setattr(
        'graph_service.sources.sync.create_graphiti_client', lambda _settings: graphiti
    )

    job = await _execute(manager, store, source['id'])

    assert job['status'] == 'succeeded'
    assert job['updated'] == 1
    call = graphiti.add_episode.await_args
    assert 'uuid' not in call.kwargs
    assert call.kwargs['previous_episode_uuids'] == ['previous-episode']
    assert call.kwargs['saga_previous_episode_uuid'] == 'previous-episode'
    item = store.get_item(source['id'], 'guide.pdf')
    assert item is not None
    assert item['episode_uuid'] == 'new-generated-episode'
    assert item['content_hash'] == new_document.content_hash
