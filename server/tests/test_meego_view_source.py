from __future__ import annotations

from typing import Any

import pytest
from fastapi import HTTPException

from graph_service.routers.sources import _normalize_config
from graph_service.sources.connectors import MeegoOAuthConnector


def test_meego_view_url_defines_exact_source_scope():
    config = _normalize_config(
        'meego',
        {
            'view_url': 'https://meego.larkoffice.com/ai_search_rec/storyView/lls2qdsvR'
        },
        meego_host='meego.larkoffice.com',
    )

    assert config == {
        'view_url': 'https://meego.larkoffice.com/ai_search_rec/storyView/lls2qdsvR',
        'project_key': 'ai_search_rec',
        'work_item_type_key': 'story',
        'view_id': 'lls2qdsvR',
        'page_size': 100,
    }


def test_meego_view_url_must_match_oauth_host():
    with pytest.raises(HTTPException, match='meego.larkoffice.com'):
        _normalize_config(
            'meego',
            {'view_url': 'https://project.feishu.cn/demo/storyView/view1'},
            meego_host='meego.larkoffice.com',
        )


def test_meego_selected_view_defines_exact_source_scope_without_url():
    config = _normalize_config(
        'meego',
        {
            'project_key': 'ai_search_rec',
            'project_name': 'AML-AI搜索',
            'work_item_type_key': 'story',
            'view_id': 'lls2qdsvR',
            'view_name': 'AI搜索引擎_20260825',
        },
        meego_host='meego.larkoffice.com',
    )

    assert config == {
        'view_url': '',
        'project_key': 'ai_search_rec',
        'project_name': 'AML-AI搜索',
        'work_item_type_key': 'story',
        'view_id': 'lls2qdsvR',
        'view_name': 'AI搜索引擎_20260825',
        'page_size': 100,
    }


class FakeMeeGoManager:
    def __init__(self):
        self.view_pages: list[int] = []

    async def meego_call(
        self,
        connection_id: str,
        resource: str,
        method: str,
        fallback: str,
        arguments: dict[str, Any],
    ) -> Any:
        assert connection_id == 'connection-1'
        if fallback == 'get_view_detail':
            page = arguments['page_num']
            self.view_pages.append(page)
            work_item_id = f'item-{page}'
            return {
                'pagination': {'page_num': page, 'page_size': 50, 'has_more': page == 1},
                'work_item_list': [
                    {
                        'work_item_attribute': {
                            'work_item_id': work_item_id,
                            'work_item_name': f'需求 {page}',
                            'update_time': f'2026-08-1{page}T00:00:00+00:00',
                            'work_item_type': {'key': 'story', 'name': '需求'},
                        },
                        'work_item_fields': [],
                    }
                ],
            }
        assert resource == 'workitem'
        assert method == 'get'
        assert fallback == 'get_workitem_brief'
        return {
            'id': arguments['work_item_id'],
            'name': f"详情 {arguments['work_item_id']}",
        }


@pytest.mark.asyncio
async def test_meego_connector_reads_only_items_returned_by_view():
    manager = FakeMeeGoManager()
    connector = MeegoOAuthConnector(
        {
            'view_url': 'https://meego.larkoffice.com/demo/storyView/view1',
            'project_key': 'demo',
            'work_item_type_key': 'story',
            'view_id': 'view1',
        },
        'connection-1',
        manager,
    )

    documents = await connector.fetch()

    assert manager.view_pages == [1, 2]
    assert [document.external_id for document in documents] == ['demo:story:view:view1']
    assert documents[0].metadata['view_id'] == 'view1'
    assert documents[0].metadata['item_count'] == 2
    assert '需求 1' in documents[0].content
    assert '需求 2' in documents[0].content


@pytest.mark.asyncio
async def test_meego_connector_reads_selected_view_without_url():
    manager = FakeMeeGoManager()
    connector = MeegoOAuthConnector(
        {
            'project_key': 'demo',
            'work_item_type_key': 'story',
            'view_id': 'view1',
        },
        'connection-1',
        manager,
    )

    documents = await connector.fetch()

    assert manager.view_pages == [1, 2]
    assert [document.external_id for document in documents] == ['demo:story:view:view1']
