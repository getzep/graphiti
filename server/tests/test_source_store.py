from concurrent.futures import ThreadPoolExecutor

from graph_service.sources.models import utc_now_iso
from graph_service.sources.store import SourceStore


def _source(store: SourceStore) -> dict:
    return store.create_source(
        kind='local', name='离线资料', group_id='neo4j', config={}, enabled=True
    )


def test_store_updates_unchanged_item_metadata_and_reconciles_safely(tmp_path):
    store = SourceStore(tmp_path / 'state.db')
    source = _source(store)
    store.upsert_item(
        source_id=source['id'],
        external_id='guide.md',
        remote_version='1',
        content_hash='hash',
        episode_uuid='episode',
        title='旧标题',
        source_updated_at='2026-01-01T00:00:00+00:00',
    )

    scan_started_at = utc_now_iso()
    store.touch_items(source['id'], {'guide.md'})
    store.touch_item_metadata(
        source['id'],
        'guide.md',
        remote_version='2',
        title='新标题',
        source_updated_at='2026-01-02T00:00:00+00:00',
    )

    assert store.mark_items_missing(source['id'], scan_started_at) == 0
    item = store.get_item(source['id'], 'guide.md')
    assert item is not None
    assert item['remote_version'] == '2'
    assert item['title'] == '新标题'
    assert item['source_updated_at'] == '2026-01-02T00:00:00+00:00'
    assert item['deleted_at'] is None


def test_create_job_coalesces_concurrent_requests(tmp_path):
    store = SourceStore(tmp_path / 'state.db')
    source = _source(store)

    def create_job(index: int) -> dict:
        return store.create_job(source['id'], full_sync=index % 2 == 0)

    with ThreadPoolExecutor(max_workers=8) as executor:
        jobs = list(executor.map(create_job, range(24)))

    assert len({job['id'] for job in jobs}) == 1
    assert len(store.list_jobs()) == 1
    assert store.stats()['active_jobs'] == 1
