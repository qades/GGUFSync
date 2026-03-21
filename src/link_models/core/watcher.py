"""Filesystem watcher with download detection."""

from __future__ import annotations

import asyncio
import time
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

from watchdog.events import (
    FileSystemEvent,
    FileSystemEventHandler,
    FileCreatedEvent,
    FileModifiedEvent,
    FileDeletedEvent,
    FileMovedEvent,
    DirCreatedEvent,
    DirDeletedEvent,
    DirMovedEvent,
)
from watchdog.observers import Observer

from .constants import (
    DOWNLOAD_CHECK_INTERVAL,
    DOWNLOAD_STABLE_COUNT,
    DOWNLOAD_MAX_WAIT,
    GGUF_EXTENSION,
)
from .exceptions import WatchError
from .logging import get_logger
from .models import SyncEvent, SyncEventType, is_partial_download, get_real_filename

logger = get_logger(__name__)

# Type alias for event handler
EventHandler = Callable[[SyncEvent], Any]


@dataclass
class PendingDownload:
    """Tracks a file that appears to be downloading."""
    path: Path
    first_seen: float = field(default_factory=time.time)
    last_check: float = field(default_factory=time.time)
    last_size: int = 0
    stable_count: int = 0
    real_name: str | None = None
    is_partial: bool = False  # True if this was originally a partial download file


class DownloadDetector:
    """Detects when file downloads are complete."""
    
    def __init__(
        self,
        check_interval: float = DOWNLOAD_CHECK_INTERVAL,
        stable_count: int = DOWNLOAD_STABLE_COUNT,
        max_wait: int = DOWNLOAD_MAX_WAIT,
    ) -> None:
        self.check_interval = check_interval
        self.stable_count_required = stable_count
        self.max_wait = max_wait
        self._pending: dict[Path, PendingDownload] = {}
    
    def is_partial(self, path: Path) -> bool:
        """Check if path indicates a partial download."""
        return is_partial_download(path.name)
    
    def get_real_name(self, path: Path) -> str:
        """Get the real filename without partial extensions."""
        if self.is_partial(path):
            return get_real_filename(path.name)
        return path.name
    
    def add_pending(self, path: Path) -> PendingDownload:
        """Add a file to pending downloads.
        
        Args:
            path: Path to the downloading file
            
        Returns:
            PendingDownload tracker
        """
        pending = PendingDownload(
            path=path,
            real_name=self.get_real_name(path),
            is_partial=True,  # This is a partial download file
        )
        self._pending[path] = pending
        logger.debug(
            "Added pending download",
            path=path.name,
            real_name=pending.real_name,
        )
        return pending
    
    def remove_pending(self, path: Path) -> PendingDownload | None:
        """Remove a file from pending downloads.
        
        Args:
            path: Path to remove
            
        Returns:
            The removed PendingDownload or None
        """
        return self._pending.pop(path, None)
    
    def check_complete(self, path: Path) -> tuple[bool, Path | None]:
        """Check if a download is complete.
        
        Args:
            path: Path to check
            
        Returns:
            Tuple of (is_complete, final_path)
            - is_complete: True if download finished
            - final_path: The path to the completed file (may be different for partials)
        """
        now = time.time()
        
        # Check if this is a tracked partial download
        if path in self._pending:
            pending = self._pending[path]
            
            # Check timeout
            if now - pending.first_seen > self.max_wait:
                logger.warning("Download timed out", path=str(path))
                self._pending.pop(path, None)
                return True, path
            
            # For partial files, check if real file appeared
            if pending.real_name and pending.real_name != path.name:
                real_path = path.parent / pending.real_name
                if real_path.exists():
                    # Partial file was renamed to real file
                    self._pending.pop(path, None)
                    return True, real_path
            
            # Check if file still exists and get size (atomic check)
            try:
                current_size = path.stat().st_size
            except FileNotFoundError:
                self._pending.pop(path, None)
                return False, None
            except OSError:
                self._pending.pop(path, None)
                return False, None
            
            # Check if size stabilized
            # Non-partial files need fewer checks since they're likely complete
            required_stable = 1 if not pending.is_partial else self.stable_count_required
            
            if current_size == pending.last_size:
                pending.stable_count += 1
                if pending.stable_count >= required_stable:
                    self._pending.pop(path, None)
                    return True, path
            else:
                pending.stable_count = 0
                pending.last_size = current_size
            
            pending.last_check = now
            return False, None
        
        # Check an untracked file (not a partial download)
        try:
            current_size = path.stat().st_size
        except FileNotFoundError:
            return False, None
        except OSError:
            return False, None
        
        # For non-partial files, use lower stable count threshold
        # since they're likely already complete (moved/renamed)
        pending = PendingDownload(
            path=path,
            last_size=current_size,
            stable_count=1,  # Start at 1 since we just checked it's stable
            is_partial=False,
        )
        self._pending[path] = pending
        return False, None
    
    def check_all_pending(self) -> list[tuple[Path, Path]]:
        """Check all pending downloads and return completed ones.
        
        Returns:
            List of (original_path, final_path) for completed downloads
        """
        completed = []
        to_remove = []
        
        for path, pending in list(self._pending.items()):
            is_complete, final_path = self.check_complete(path)
            if is_complete and final_path:
                completed.append((path, final_path))
                to_remove.append(path)
        
        for path in to_remove:
            self._pending.pop(path, None)
        
        return completed
    
    @property
    def pending_count(self) -> int:
        """Number of pending downloads."""
        return len(self._pending)

    def get_pending_paths(self) -> set[Path]:
        """Get set of all pending download paths (including real names)."""
        paths = set(self._pending.keys())
        for pending in self._pending.values():
            if pending.real_name:
                paths.add(pending.path.parent / pending.real_name)
        return paths


class ModelEventHandler(FileSystemEventHandler):
    """Watchdog event handler for model files."""
    
    def __init__(
        self,
        callback: EventHandler,
        source_dirs: list[Path],
        download_detector: DownloadDetector,
    ) -> None:
        self.callback = callback
        self.source_dirs = [d.resolve() for d in source_dirs]
        self.download_detector = download_detector
    
    def _get_source_dir(self, path: Path) -> Path | None:
        """Determine which source directory a path belongs to."""
        try:
            resolved = path.resolve()
            for source_dir in self.source_dirs:
                # Use is_relative_to for proper path comparison (handles spaces correctly)
                try:
                    if resolved.is_relative_to(source_dir):
                        return source_dir
                except AttributeError:
                    # Python < 3.9 fallback
                    try:
                        resolved.relative_to(source_dir)
                        return source_dir
                    except ValueError:
                        continue
        except (OSError, ValueError) as e:
            logger.debug("Failed to resolve path", path=str(path), error=str(e))
        return None
    
    def _is_gguf(self, path: Path) -> bool:
        """Check if file is a GGUF file."""
        return path.suffix.lower() == GGUF_EXTENSION
    
    def _handle_file_event(
        self,
        event: FileSystemEvent,
        event_type: SyncEventType,
    ) -> None:
        """Handle a file event."""
        path = Path(event.src_path)
        
        logger.debug(
            "Handling file event",
            event_type=event_type.name,
            path=str(path),
            filename=path.name,
        )
        
        # Only care about GGUF files
        if not self._is_gguf(path):
            logger.debug("Skipping non-GGUF file", filename=path.name)
            return
        
        source_dir = self._get_source_dir(path)
        if not source_dir:
            logger.debug("File not in watched directories", path=str(path))
            return
        
        logger.debug(
            "Processing GGUF file",
            filename=path.name,
            source_dir=str(source_dir),
        )
        
        # Check if this is a partial download
        is_partial = self.download_detector.is_partial(path)
        
        if is_partial:
            # Track the partial download
            self.download_detector.add_pending(path)
            logger.debug("Added partial download to tracking", filename=path.name)
            return
        
        # For new/modified files, check if download is complete
        if event_type in (SyncEventType.FILE_CREATED, SyncEventType.FILE_MODIFIED):
            is_complete, final_path = self.download_detector.check_complete(path)
            if not is_complete:
                # Still downloading, will be picked up by polling
                logger.debug("File still downloading", filename=path.name)
                return
            path = final_path or path
        
        sync_event = SyncEvent(
            event_type=event_type,
            path=path,
            source_dir=source_dir,
            is_partial=False,
        )
        
        logger.debug(
            "Calling callback for event",
            event_type=event_type.name,
            filename=path.name,
        )
        
        try:
            self.callback(sync_event)
        except Exception as e:
            logger.error("Error handling event", error=str(e), event=str(sync_event))
    
    def on_created(self, event: FileSystemEvent) -> None:
        """Handle file/directory creation."""
        if event.is_directory:
            logger.debug("Ignoring directory creation", path=event.src_path)
            return
        logger.debug("File created event", path=event.src_path)
        self._handle_file_event(event, SyncEventType.FILE_CREATED)
    
    def on_modified(self, event: FileSystemEvent) -> None:
        """Handle file modification."""
        if event.is_directory:
            return
        logger.debug("File modified event", path=event.src_path)
        self._handle_file_event(event, SyncEventType.FILE_MODIFIED)
    
    def on_deleted(self, event: FileSystemEvent) -> None:
        """Handle file/directory deletion."""
        if event.is_directory:
            return
        
        path = Path(event.src_path)
        logger.debug("File deleted event", path=str(path))
        
        if not self._is_gguf(path):
            logger.debug("Skipping non-GGUF deletion", filename=path.name)
            return
        
        source_dir = self._get_source_dir(path)
        if not source_dir:
            logger.debug("Deleted file not in watched directories", path=str(path))
            return
        
        # Remove from pending if tracked
        self.download_detector.remove_pending(path)
        
        sync_event = SyncEvent(
            event_type=SyncEventType.FILE_DELETED,
            path=path,
            source_dir=source_dir,
            is_partial=False,
        )
        
        try:
            self.callback(sync_event)
        except Exception as e:
            logger.error("Error handling deletion", error=str(e), path=str(path))
    
    def on_moved(self, event: FileSystemEvent) -> None:
        """Handle file/directory move/rename."""
        if event.is_directory:
            return
        
        src_path = Path(event.src_path)
        dest_path = Path(event.dest_path)
        
        logger.debug(
            "Handling move event",
            src=str(src_path),
            dest=str(dest_path),
            src_name=src_path.name,
            dest_name=dest_path.name,
        )
        
        # Check if this is a partial download completing (any .part/.tmp -> .gguf)
        # This handles cases where the rename happens before we track the partial
        src_is_partial = self.download_detector.is_partial(src_path)
        dest_is_gguf = self._is_gguf(dest_path)
        
        if src_is_partial and dest_is_gguf:
            # A partial file was renamed to a GGUF file - always treat as completed download
            # Remove from pending if we were tracking it
            self.download_detector.remove_pending(src_path)
            
            source_dir = self._get_source_dir(dest_path)
            if source_dir:
                logger.info(
                    "Partial download completed (renamed)",
                    src=src_path.name,
                    dest=dest_path.name,
                )
                sync_event = SyncEvent(
                    event_type=SyncEventType.DOWNLOAD_COMPLETED,
                    path=dest_path,
                    source_dir=source_dir,
                    is_partial=False,
                )
                try:
                    self.callback(sync_event)
                except Exception as e:
                    logger.error("Error handling completion", error=str(e))
            return
        
        # Check if a non-partial file was moved and became a GGUF file
        # (e.g., moved from a staging directory)
        if dest_is_gguf and not self._is_gguf(src_path):
            logger.debug("File became GGUF after move", filename=dest_path.name)
            self.on_created(FileCreatedEvent(dest_path))
            return
        
        # Regular move between directories - treat as delete + create
        if self._is_gguf(src_path):
            logger.debug("Processing move as delete", filename=src_path.name)
            self.on_deleted(FileDeletedEvent(src_path))
        
        if self._is_gguf(dest_path):
            logger.debug("Processing move as create", filename=dest_path.name)
            self.on_created(FileCreatedEvent(dest_path))


class FileSystemWatcher:
    """Cross-platform filesystem watcher for model directories."""
    
    def __init__(
        self,
        source_dirs: list[Path],
        callback: EventHandler,
        *,
        check_interval: float = DOWNLOAD_CHECK_INTERVAL,
        stable_count: int = DOWNLOAD_STABLE_COUNT,
        max_wait: int = DOWNLOAD_MAX_WAIT,
        recursive: bool = True,
    ) -> None:
        """Initialize the filesystem watcher.
        
        Args:
            source_dirs: Directories to watch
            callback: Function to call when events occur
            check_interval: Seconds between download size checks
            stable_count: Consecutive stable checks to confirm download complete
            max_wait: Maximum seconds to wait for download
            recursive: Watch subdirectories recursively
        """
        self.source_dirs = source_dirs
        self.callback = callback
        self.recursive = recursive
        
        self.download_detector = DownloadDetector(
            check_interval=check_interval,
            stable_count=stable_count,
            max_wait=max_wait,
        )
        
        self._observer: Observer | None = None
        self._handler: ModelEventHandler | None = None
        self._running = False
        self._poll_task: asyncio.Task | None = None
    
    def start(self) -> None:
        """Start watching filesystem."""
        if self._running:
            return
        
        logger.info(
            "Starting filesystem watcher",
            dirs=[str(d) for d in self.source_dirs],
            recursive=self.recursive,
        )
        
        self._handler = ModelEventHandler(
            callback=self.callback,
            source_dirs=self.source_dirs,
            download_detector=self.download_detector,
        )
        
        self._observer = Observer()
        
        for source_dir in self.source_dirs:
            if not source_dir.exists():
                raise WatchError(f"Directory does not exist: {source_dir}")
            
            self._observer.schedule(
                self._handler,
                str(source_dir),
                recursive=self.recursive,
            )
            logger.debug("Scheduled watch", path=str(source_dir))
        
        self._observer.start()
        self._running = True
    
    def stop(self) -> None:
        """Stop watching filesystem."""
        if not self._running:
            return
        
        logger.info("Stopping filesystem watcher")
        
        if self._observer:
            self._observer.stop()
            self._observer.join()
            self._observer = None
        
        self._running = False
    
    async def run(self) -> None:
        """Run the watcher with async polling for pending downloads."""
        self.start()
        
        try:
            while self._running:
                # Poll for completed downloads
                completed = self.download_detector.check_all_pending()
                for original_path, final_path in completed:
                    source_dir = self._handler._get_source_dir(final_path)
                    if source_dir:
                        event = SyncEvent(
                            event_type=SyncEventType.DOWNLOAD_COMPLETED,
                            path=final_path,
                            source_dir=source_dir,
                            is_partial=False,
                        )
                        try:
                            self.callback(event)
                        except Exception as e:
                            logger.error("Error handling completion", error=str(e))
                
                await asyncio.sleep(self.download_detector.check_interval)
        
        except asyncio.CancelledError:
            logger.info("Watcher cancelled")
        finally:
            self.stop()
    
    def __enter__(self) -> FileSystemWatcher:
        """Context manager entry."""
        self.start()
        return self
    
    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        """Context manager exit."""
        self.stop()
