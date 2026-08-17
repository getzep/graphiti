from __future__ import annotations

import asyncio
import logging
from datetime import timezone
from pathlib import Path
from typing import Any

from graphiti_core.nodes import EpisodeType

from graph_service.config import Settings
from graph_service.wikis.templates import (
    PROJECT_WIKI_ENTITY_TYPES,
    PROJECT_WIKI_EXTRACTION_INSTRUCTIONS,
)
from graph_service.zep_graphiti import ZepGraphiti, create_graphiti_client

from .connectors import ConnectorError, build_connector
from .models import SourceDocument, utc_now_iso
from .store import SourceStore

logger = logging.getLogger(__name__)


class SyncManager:
    """Run persistent, observable source synchronization jobs."""

    def __init__(
        self,
        settings: Settings,
        store: SourceStore,
        *,
        connection_manager: Any | None = None,
    ):
        self.settings = settings
        self.store = store
        self.connection_manager = connection_manager
        self.upload_root = Path(settings.upload_root)
        self.upload_root.mkdir(parents=True, exist_ok=True)
        self._semaphore = asyncio.Semaphore(settings.sync_concurrency)
        self._group_locks: dict[str, asyncio.Lock] = {}
        self._tasks: dict[str, asyncio.Task[None]] = {}
        self.store.fail_interrupted_jobs()

    def enqueue(self, source_id: str, *, full_sync: bool = False) -> dict[str, Any]:
        source = self.store.get_source(source_id)
        if not source['enabled']:
            raise ConnectorError('该数据源已停用')
        active = self.store.active_job_for_source(source_id)
        if active:
            return active
        job = self.store.create_job(source_id, full_sync=full_sync)
        if job['id'] in self._tasks:
            return job
        task = asyncio.create_task(self._run(job['id']), name=f'source-sync-{job["id"]}')
        self._tasks[job['id']] = task
        task.add_done_callback(lambda done: self._task_done(job['id'], done))
        return job

    def _task_done(self, job_id: str, task: asyncio.Task[None]) -> None:
        self._tasks.pop(job_id, None)
        if not task.cancelled() and (error := task.exception()) is not None:
            logger.error(
                'Source sync task %s crashed: %s',
                job_id,
                error,
                exc_info=(type(error), error, error.__traceback__),
            )

    async def shutdown(self) -> None:
        tasks = list(self._tasks.values())
        if not tasks:
            return
        for task in tasks:
            task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)

    @staticmethod
    def _error(exc: BaseException) -> str:
        return str(exc).replace('\n', ' ')[:1000]

    async def _add_document(
        self,
        graphiti: ZepGraphiti,
        source: dict[str, Any],
        document: SourceDocument,
        previous_episode_uuid: str | None,
    ) -> str:
        kwargs: dict[str, Any] = {}
        if previous_episode_uuid:
            kwargs['previous_episode_uuids'] = [previous_episode_uuid]
            kwargs['saga_previous_episode_uuid'] = previous_episode_uuid
        result = await graphiti.add_episode(
            group_id=source['group_id'],
            name=document.title[:500],
            episode_body=document.episode_body(source['kind']),
            reference_time=document.updated_at.astimezone(timezone.utc),
            source=EpisodeType.text,
            source_description=(f'{source["kind"]}:{source["id"]}:{document.external_id}'[:500]),
            saga=f'source:{source["id"]}:{document.external_id}'[:500],
            entity_types=PROJECT_WIKI_ENTITY_TYPES,
            custom_extraction_instructions=PROJECT_WIKI_EXTRACTION_INSTRUCTIONS,
            **kwargs,
        )
        # Graphiti interprets a supplied UUID as an already-persisted episode to reprocess.
        # New source versions must let Graphiti create the node and persist its returned UUID.
        return str(result.episode.uuid)

    async def _run(self, job_id: str) -> None:
        async with self._semaphore:
            job = self.store.get_job(job_id)
            source = self.store.get_source(job['source_id'])
            group_lock = self._group_locks.setdefault(source['group_id'], asyncio.Lock())
            async with group_lock:
                await self._execute(job, source)

    async def _execute(self, job: dict[str, Any], source: dict[str, Any]) -> None:
        job_id = job['id']
        scan_started_at = utc_now_iso()
        counts = {'scanned': 0, 'created': 0, 'updated': 0, 'skipped': 0, 'failed': 0}
        graphiti: ZepGraphiti | None = None
        connector = None
        self.store.update_job(job_id, status='running', started_at=scan_started_at)
        self.store.set_source_state(source['id'], status='syncing', last_error=None)

        try:
            connector = build_connector(
                source,
                self.settings,
                self.upload_root,
                self.connection_manager,
            )
            documents = await connector.fetch(
                watermark_ms=source.get('watermark_ms'), full_sync=job['full_sync']
            )
            # The connector inventory can observe an item even when its latest body fails to
            # download or parse. Touch those rows before reconciliation so transient failures
            # never masquerade as remote deletions.
            self.store.touch_items(source['id'], connector.seen_external_ids)
            counts['failed'] += len(connector.errors)
            max_watermark = source.get('watermark_ms') or 0

            for document in documents:
                counts['scanned'] += 1
                # Persist discovery before the expensive Graphiti extraction so the UI does
                # not look stuck at 0 while a document is being processed by the model.
                self.store.update_job(
                    job_id, **counts, warnings=connector.warnings + connector.errors
                )
                existing = self.store.get_item(source['id'], document.external_id)
                if not document.content.strip():
                    counts['failed'] += 1
                    connector.errors.append(f'{document.title}: 内容为空，已跳过')
                    self.store.update_job(
                        job_id, **counts, warnings=connector.warnings + connector.errors
                    )
                    continue
                if existing and existing['content_hash'] == document.content_hash:
                    counts['skipped'] += 1
                    self.store.touch_item_metadata(
                        source['id'],
                        document.external_id,
                        remote_version=document.remote_version,
                        title=document.title,
                        source_updated_at=document.updated_at.isoformat(),
                    )
                else:
                    if not self.settings.openai_api_key:
                        raise ConnectorError('请先设置 OPENAI_API_KEY/ARK_API_KEY 再执行图谱抽取')
                    if graphiti is None:
                        graphiti = create_graphiti_client(self.settings)
                    try:
                        episode_uuid = await self._add_document(
                            graphiti,
                            source,
                            document,
                            existing['episode_uuid'] if existing else None,
                        )
                    except Exception as exc:
                        counts['failed'] += 1
                        connector.errors.append(f'{document.title}: {self._error(exc)}')
                        logger.exception(
                            'Failed to ingest source document %s', document.external_id
                        )
                    else:
                        counts['updated' if existing else 'created'] += 1
                        self.store.upsert_item(
                            source_id=source['id'],
                            external_id=document.external_id,
                            remote_version=document.remote_version,
                            content_hash=document.content_hash,
                            episode_uuid=episode_uuid,
                            title=document.title,
                            source_updated_at=document.updated_at.isoformat(),
                        )
                max_watermark = max(
                    max_watermark,
                    int(document.updated_at.timestamp() * 1000),
                )
                self.store.update_job(
                    job_id, **counts, warnings=connector.warnings + connector.errors
                )

            complete_inventory = connector.inventory_complete and (
                source['kind'] in {'local', 'feishu'} or job['full_sync']
            )
            if complete_inventory:
                deleted = self.store.mark_items_missing(source['id'], scan_started_at)
                if deleted:
                    connector.warnings.append(
                        f'{deleted} 个远端已移除条目已标记 tombstone；历史图谱仍保留'
                    )

            finished_at = utc_now_iso()
            status = 'succeeded' if counts['failed'] == 0 else 'partial'
            self.store.update_job(
                job_id,
                status=status,
                finished_at=finished_at,
                warnings=connector.warnings + connector.errors,
                **counts,
            )
            self.store.set_source_state(
                source['id'],
                status='idle' if status == 'succeeded' else 'error',
                last_error=None if status == 'succeeded' else '部分条目同步失败',
                last_sync_at=finished_at,
                watermark_ms=max_watermark if counts['failed'] == 0 else None,
            )
        except asyncio.CancelledError:
            finished_at = utc_now_iso()
            self.store.update_job(
                job_id,
                status='failed',
                error='任务被取消',
                finished_at=finished_at,
                **counts,
            )
            self.store.set_source_state(source['id'], status='error', last_error='任务被取消')
            raise
        except Exception as exc:
            error = self._error(exc)
            finished_at = utc_now_iso()
            warnings = connector.warnings + connector.errors if connector else []
            self.store.update_job(
                job_id,
                status='failed',
                error=error,
                warnings=warnings,
                finished_at=finished_at,
                **counts,
            )
            self.store.set_source_state(source['id'], status='error', last_error=error)
            logger.exception('Source sync job %s failed', job_id)
        finally:
            if source.get('wiki_id'):
                wiki = self.store.refresh_wiki_build_status(source['wiki_id'])
                if wiki['candidate_status'] == 'ready':
                    self.store.publish_wiki(source['wiki_id'])
            if graphiti is not None:
                try:
                    await graphiti.close()
                except Exception:
                    logger.exception('Failed to close Graphiti client for source job %s', job_id)
