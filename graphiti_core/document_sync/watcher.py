"""File watching with hourly batched sync.

This module implements file system monitoring for the corpus directory using the Watchdog
library. It queues document changes and processes them in hourly batches to prevent
graph noise from rapid edits.

Architecture:
    Observer Thread (Watchdog):
        - Monitors corpus directory recursively for .md files
        - Detects modification and rename/move events
        - Thread-safe queue operations using threading.Lock

    Async Sync Task:
        - Sleeps for configured interval (default 1 hour)
        - Processes rename queue first (updates metadata URIs)
        - Processes change queue (syncs modified documents)
        - Filters renamed destination files from change queue to prevent duplicates

    Thread Safety:
        - Watchdog runs in separate thread
        - Queues protected by threading.Lock
        - Event loop integration via loop.create_task()

Manual Sync:
    Call trigger_sync() to immediately process queued changes without waiting
    for the next scheduled batch.
"""

import asyncio
import logging
import threading
from pathlib import Path

from watchdog.events import FileModifiedEvent, FileMovedEvent, FileSystemEventHandler
from watchdog.observers import Observer

from .sync_manager import DocumentSyncManager

logger = logging.getLogger(__name__)


class CorpusFileHandler(FileSystemEventHandler):
    """Handles file system events for the corpus directory."""

    def __init__(self, watcher: 'DocumentWatcher'):
        """Initialize file handler.

        Args:
            watcher: Parent DocumentWatcher instance
        """
        self.watcher = watcher

    def on_modified(self, event):
        """Handle file modification events.

        Args:
            event: File modification event
        """
        if event.is_directory:
            return

        file_path = Path(event.src_path)

        # Only watch markdown files
        if file_path.suffix == '.md':
            self.watcher._on_modified(file_path)

    def on_moved(self, event):
        """Handle file rename/move events.

        Args:
            event: File move event
        """
        if event.is_directory:
            return

        src_path = Path(event.src_path)
        dest_path = Path(event.dest_path)

        # Only watch markdown files
        if src_path.suffix == '.md' or dest_path.suffix == '.md':
            self.watcher._on_moved(src_path, dest_path)


class DocumentWatcher:
    """Watches corpus directory and batches document syncs."""

    def __init__(
        self,
        sync_manager: DocumentSyncManager,
        sync_interval_hours: float = 1.0,
    ):
        """Initialize document watcher.

        Args:
            sync_manager: DocumentSyncManager instance
            sync_interval_hours: Hours between batch syncs (default: 1.0)
        """
        self.sync_manager = sync_manager
        self.sync_interval = sync_interval_hours * 3600  # Convert to seconds
        self.observer = Observer()
        self.change_queue: set[Path] = set()
        self.rename_queue: list[tuple[str, str]] = []  # (old_uri, new_uri) pairs
        self.loop: asyncio.AbstractEventLoop | None = None
        self.sync_task: asyncio.Task | None = None
        self._lock = threading.Lock()
        self._running = False

    def start(self, loop: asyncio.AbstractEventLoop):
        """Start file watcher and background sync task.

        Args:
            loop: Event loop to run sync task in
        """
        if self._running:
            logger.warning('Document watcher already running')
            return

        self.loop = loop
        self._running = True

        # Create event handler
        event_handler = CorpusFileHandler(self)

        # Watch corpus directory recursively
        self.observer.schedule(
            event_handler,
            str(self.sync_manager.corpus_path),
            recursive=True,
        )

        # Start observer thread
        self.observer.start()

        # Schedule periodic sync task
        self.sync_task = loop.create_task(self._periodic_sync())

    async def stop(self):
        """Stop observer and cancel sync task."""
        if not self._running:
            return

        self._running = False

        # Stop observer
        self.observer.stop()
        self.observer.join()

        # Cancel sync task
        if self.sync_task:
            self.sync_task.cancel()
            try:
                await self.sync_task
            except asyncio.CancelledError:
                pass

    def _on_modified(self, file_path: Path):
        """Thread-safe: Add file to change queue.

        Called from watchdog thread.

        Args:
            file_path: Path to modified file
        """
        with self._lock:
            self.change_queue.add(file_path)

    def _on_moved(self, old_path: Path, new_path: Path):
        """Thread-safe: Add rename to queue.

        Called from watchdog thread.

        Args:
            old_path: Previous file path
            new_path: New file path
        """
        with self._lock:
            # Convert to URIs relative to corpus
            old_uri = str(old_path.relative_to(self.sync_manager.corpus_path))
            new_uri = str(new_path.relative_to(self.sync_manager.corpus_path))
            self.rename_queue.append((old_uri, new_uri))

    async def _periodic_sync(self):
        """Background task: sleep → process queues → repeat."""
        try:
            while self._running:
                await asyncio.sleep(self.sync_interval)

                if not self._running:
                    break

                await self._process_queues()

        except asyncio.CancelledError:
            raise

    async def _process_queues(self):
        """Process rename queue first, then change queue."""
        # Pop from queues (thread-safe)
        with self._lock:
            renames = list(self.rename_queue)
            changes = set(self.change_queue)
            self.rename_queue.clear()
            self.change_queue.clear()

        if not renames and not changes:
            return

        # Process renames first (updates metadata URIs)
        for old_uri, new_uri in renames:
            await self.sync_manager.handle_rename(old_uri, new_uri)

        # Remove renamed destination files from change queue
        # to prevent them from being synced as new files
        renamed_dest_paths = {self.sync_manager.corpus_path / new_uri for _, new_uri in renames}
        changes = changes - renamed_dest_paths

        # Process changes
        for file_path in changes:
            if not file_path.exists():
                continue

            await self.sync_manager.sync_document(file_path)

    async def trigger_sync(self):
        """Manually trigger sync of queued changes."""
        await self._process_queues()
