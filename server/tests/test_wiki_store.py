import pytest

from graph_service.routers.wikis import wiki_mcp_url
from graph_service.sources.models import utc_now_iso
from graph_service.sources.store import SourceStore
from graph_service.wikis.templates import project_wiki_plan


def test_wikis_use_distinct_candidate_namespaces_and_publish_by_pointer(tmp_path):
    store = SourceStore(tmp_path / 'state.db')
    first = store.create_wiki(name='Product Wiki', slug='product-wiki')
    second = store.create_wiki(name='Support Wiki', slug='support-wiki')

    assert first['candidate_group_id'] != second['candidate_group_id']
    assert first['published_group_id'] is None

    source = store.create_source(
        kind='local',
        name='Docs',
        wiki_id=first['id'],
        group_id=first['candidate_group_id'],
        config={},
        enabled=True,
    )
    candidate, source_ids = store.prepare_wiki_build(first['id'])

    assert source_ids == [source['id']]
    assert candidate['candidate_group_id'] != first['candidate_group_id']
    assert store.get_source(source['id'])['group_id'] == candidate['candidate_group_id']

    store.set_source_state(source['id'], status='idle', last_sync_at=utc_now_iso())
    assert store.refresh_wiki_build_status(first['id'])['candidate_status'] == 'ready'

    published = store.publish_wiki(first['id'])
    assert published['published_group_id'] == candidate['candidate_group_id']
    assert store.get_published_group(first['id']) == candidate['candidate_group_id']
    assert store.source_can_sync(store.get_source(source['id'])) is False

    next_candidate, _ = store.prepare_wiki_build(first['id'])
    assert next_candidate['candidate_group_id'] != published['published_group_id']
    assert store.source_can_sync(store.get_source(source['id'])) is True


def test_wiki_cannot_publish_before_a_successful_build(tmp_path):
    store = SourceStore(tmp_path / 'state.db')
    wiki = store.create_wiki(name='Empty Wiki')

    with pytest.raises(ValueError, match='Candidate'):
        store.publish_wiki(wiki['id'])
    with pytest.raises(ValueError, match='数据源'):
        store.prepare_wiki_build(wiki['id'])


def test_wiki_slug_is_unique(tmp_path):
    store = SourceStore(tmp_path / 'state.db')
    store.create_wiki(name='One', slug='same-wiki')

    with pytest.raises(ValueError, match='slug'):
        store.create_wiki(name='Two', slug='same-wiki')


def test_wiki_keeps_goal_scope_and_versioned_project_plan(tmp_path):
    store = SourceStore(tmp_path / 'state.db')
    goal = '帮助新 PM 理解项目、产品、需求与负责人之间的关系'

    wiki = store.create_wiki(
        name='Viking AI 搜索',
        goal=goal,
        data_scope='specified',
        plan=project_wiki_plan(goal),
    )

    assert wiki['goal'] == goal
    assert wiki['data_scope'] == 'specified'
    assert wiki['plan_version'] == 1
    assert wiki['plan']['template'] == 'project_wiki'
    assert {item['key'] for item in wiki['plan']['entity_types']} >= {
        'Project',
        'Product',
        'Requirement',
        'Person',
    }


def test_wiki_mcp_url_replaces_admin_mcp_suffix():
    wiki_id = 'a' * 32

    assert (
        wiki_mcp_url('https://vaka.example.com/gateway/mcp/', wiki_id)
        == f'https://vaka.example.com/gateway/wiki/{wiki_id}/mcp'
    )


def test_wiki_jobs_can_be_limited_to_the_current_candidate_build(tmp_path):
    store = SourceStore(tmp_path / 'state.db')
    wiki = store.create_wiki(name='Product Wiki')
    source = store.create_source(
        kind='local',
        name='Docs',
        wiki_id=wiki['id'],
        group_id=wiki['candidate_group_id'],
        config={},
        enabled=True,
    )
    old_job = store.create_job(source['id'], full_sync=True)
    store.update_job(old_job['id'], status='failed', finished_at=utc_now_iso())

    candidate, _ = store.prepare_wiki_build(wiki['id'])
    current_job = store.create_job(source['id'], full_sync=True)

    jobs = store.list_wiki_jobs(
        wiki['id'], created_since=candidate['candidate_started_at']
    )

    assert [job['id'] for job in jobs] == [current_job['id']]
