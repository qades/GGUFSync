"""Synchronization engine for model management."""

from __future__ import annotations

import os
import fnmatch
from collections import defaultdict
from pathlib import Path
from typing import Any

from .gguf_parser import ParallelGGUFParser, parse_gguf_file
from .logging import get_logger, log_action
from .models import (
    AppConfig,
    ModelInfo,
    ModelGroup,
    SyncEvent,
    SyncEventType,
    SyncAction,
    get_multipart_base,
    is_partial_download,
    get_mmproj_base,
    strip_quantization_suffix,
)
from ..backends.base import Backend, BackendResult

logger = get_logger(__name__)


class ModelFilter:
    """Handles model filtering via ignore files."""

    def __init__(self, patterns: list[str] | None = None) -> None:
        """Initialize filter with patterns.

        Args:
            patterns: List of glob patterns to ignore
        """
        self.patterns = patterns or []

    def load_from_file(self, path: Path) -> None:
        """Load ignore patterns from file.

        Args:
            path: Path to ignore file
        """
        if not path.exists():
            return

        try:
            with open(path, "r", encoding="utf-8") as f:
                self.patterns = [
                    line.strip() for line in f if line.strip() and not line.strip().startswith("#")
                ]
            logger.debug("Loaded ignore patterns", path=str(path), count=len(self.patterns))
        except OSError as e:
            logger.warning("Failed to load ignore file", path=str(path), error=str(e))

    def should_ignore(self, model_id: str) -> bool:
        """Check if a model should be ignored.

        Args:
            model_id: Model ID to check

        Returns:
            True if model should be ignored
        """
        for pattern in self.patterns:
            if fnmatch.fnmatch(model_id, pattern):
                logger.debug("Model ignored by pattern", model_id=model_id, pattern=pattern)
                return True
            if fnmatch.fnmatch(model_id.lower(), pattern.lower()):
                logger.debug(
                    "Model ignored by pattern (case-insensitive)",
                    model_id=model_id,
                    pattern=pattern,
                )
                return True
        return False


class SyncEngine:
    """Main synchronization engine for model management."""

    def __init__(self, config: AppConfig, backends: list[Backend]) -> None:
        """Initialize sync engine.

        Args:
            config: Application configuration
            backends: List of backends to sync to
        """
        self.config = config
        self.backends = backends
        self.source_dir = config.source_dir

        # Index of all known models
        self._model_index: dict[str, ModelInfo] = {}
        self._group_index: dict[str, ModelGroup] = {}
        self._parser: ParallelGGUFParser | None = None

        # Initialize global filter
        self._global_filter = ModelFilter()
        if config.sync.global_ignore_file:
            self._global_filter.load_from_file(config.sync.global_ignore_file)

        # Per-backend filters
        self._backend_filters: dict[str, ModelFilter] = {}

    def _get_backend_filter(self, backend: Backend) -> ModelFilter:
        """Get or create filter for a backend.

        Args:
            backend: Backend to get filter for

        Returns:
            ModelFilter for the backend
        """
        backend_name = backend.name.lower().replace(" ", "_")

        if backend_name not in self._backend_filters:
            filter = ModelFilter()

            # Check for backend-specific ignore file in output directory
            ignore_file = backend.output_dir / ".linkmodelsignore"
            if ignore_file.exists():
                filter.load_from_file(ignore_file)

            # Also check backend config for ignore_file path
            if hasattr(backend, "config") and hasattr(backend.config, "ignore_file"):
                if backend.config.ignore_file:
                    filter.load_from_file(backend.config.ignore_file)

            self._backend_filters[backend_name] = filter

        return self._backend_filters[backend_name]

    def _get_context_size(self, backend: Backend, metadata: GGUFMetadata | None) -> int | None:
        """Get context size for a backend, respecting config overrides.

        Priority: backend.config.context_size > sync.default_context_size > metadata

        Args:
            backend: Backend to get context size for
            metadata: Model metadata (may contain context_length)

        Returns:
            Context size to use, or None to let backend decide
        """
        # Check backend-specific override first
        if hasattr(backend.config, "context_size") and backend.config.context_size is not None:
            return backend.config.context_size

        # Check global default
        if self.config.sync.default_context_size is not None:
            return self.config.sync.default_context_size

        # Fall back to metadata
        if metadata and metadata.context_length:
            return metadata.context_length

        # None means "use backend default"
        return None

    def _should_skip_backend(self, backend: Backend, model_id: str) -> bool:
        """Check if a model should be skipped for a specific backend.

        Args:
            backend: Backend to check
            model_id: Model ID to check

        Returns:
            True if model should be skipped for this backend
        """
        # Check global filter first
        if self._global_filter.should_ignore(model_id):
            return True

        # Check backend-specific filter
        backend_filter = self._get_backend_filter(backend)
        if backend_filter.should_ignore(model_id):
            return True

        return False

    def _backends_need_metadata(self) -> bool:
        """Check if any backend requires GGUF metadata parsing.

        Returns:
            True if metadata parsing is needed for any backend
        """
        for backend in self.backends:
            backend_name = backend.name.lower()
            if backend_name == "localai":
                # Check if LocalAI has YAML generation enabled
                from ..backends.localai import LocalAIBackend

                if isinstance(backend, LocalAIBackend):
                    if hasattr(backend, "localai_config") and backend.localai_config.generate_yaml:
                        return True
            elif backend_name == "lm studio":
                # Check if LM Studio has manifest generation enabled
                from ..backends.lmstudio import LMStudioBackend

                if isinstance(backend, LMStudioBackend):
                    if (
                        hasattr(backend, "lmstudio_config")
                        and backend.lmstudio_config.generate_manifest
                    ):
                        return True
        return False

    def _file_needs_sync(self, model_info: ModelInfo) -> bool:
        """Check if a file needs to be synced to any backend.

        Compares source file with target files across all backends.

        Args:
            model_info: Model file information

        Returns:
            True if the file needs syncing (target doesn't exist or is older)
        """
        source_mtime = model_info.mtime
        source_size = model_info.file_size

        for backend in self.backends:
            # Check if target exists and is up to date
            target_dir = backend.output_dir / model_info.path.stem
            target_file = target_dir / model_info.path.name

            try:
                if not target_file.exists():
                    return True

                target_stat = target_file.stat()
                # Check if source is newer or different size
                if source_mtime > target_stat.st_mtime:
                    return True
                if source_size != target_stat.st_size:
                    return True
            except OSError:
                # If we can't stat the target, assume it needs sync
                return True

        return False

    def setup(self) -> None:
        """Setup the sync engine and all backends."""
        logger.info("Setting up sync engine", source_dir=str(self.source_dir))

        # Ensure source directory exists
        if not self.source_dir.exists():
            raise RuntimeError(f"Source directory does not exist: {self.source_dir}")

        # Setup all backends
        for backend in self.backends:
            backend.setup()

    def full_sync(self) -> dict[str, BackendResult]:
        """Perform a full synchronization.

        Returns:
            Dictionary mapping backend names to results
        """
        logger.info("Starting full synchronization", add_only=self.config.sync.add_only)

        # Build model index
        self._build_index()

        # Build model groups
        self._build_groups()

        # Sync each group to each backend
        results = {}
        for backend in self.backends:
            result = self._sync_to_backend(backend)
            results[backend.name] = result

        # Cleanup orphans (skip in add-only mode)
        if not self.config.sync.add_only and not self.config.sync.preserve_orphans:
            for backend in self.backends:
                if hasattr(backend, "cleanup_orphans"):
                    valid_ids = set(self._group_index.keys())
                    backend.cleanup_orphans(valid_ids)
        elif self.config.sync.add_only:
            logger.info("Skipping orphan cleanup in add-only mode")

        logger.info("Full synchronization complete", backends=list(results.keys()))
        return results

    def handle_event(self, event: SyncEvent) -> dict[str, BackendResult]:
        """Handle a filesystem event.

        Args:
            event: Sync event to handle

        Returns:
            Dictionary mapping backend names to results
        """
        logger.info(
            "Handling event",
            event_type=event.event_type.name,
            path=str(event.path),
            filename=event.path.name,
            source_dir=str(event.source_dir),
        )

        results = {}

        if event.event_type == SyncEventType.FILE_DELETED:
            results = self._handle_deletion(event)
        elif event.event_type in (
            SyncEventType.FILE_CREATED,
            SyncEventType.FILE_MODIFIED,
            SyncEventType.DOWNLOAD_COMPLETED,
        ):
            results = self._handle_creation(event)

        return results

    def _build_index(self) -> None:
        """Build index of all model files.

        Optimized to only parse GGUF metadata when:
        1. A backend requires it for config generation (LocalAI YAML, LM Studio manifest)
        2. The file actually needs to be synced (target is missing or outdated)
        """
        logger.debug("Building model index")

        self._model_index = {}

        # Scan source directory
        for root, _dirs, files in os.walk(self.source_dir):
            root_path = Path(root)
            for filename in files:
                if not filename.endswith(".gguf"):
                    continue

                if is_partial_download(filename):
                    continue

                file_path = root_path / filename

                try:
                    stat = file_path.stat()
                    model_info = ModelInfo(
                        path=file_path,
                        file_size=stat.st_size,
                        mtime=stat.st_mtime,
                    )
                    self._model_index[filename] = model_info
                except OSError as e:
                    logger.warning("Failed to stat file", path=str(file_path), error=str(e))

        # Determine which files need metadata parsing
        needs_metadata = self._backends_need_metadata()

        if needs_metadata:
            # Parse metadata for all files (backend needs it for config generation)
            logger.debug("Parsing metadata for all files (backend requires it)")
            gguf_files = [m.path for m in self._model_index.values() if m.is_gguf]
        else:
            # Only parse metadata for files that need syncing
            # (for display_name extraction in models.ini)
            files_needing_sync = [
                m for m in self._model_index.values() if m.is_gguf and self._file_needs_sync(m)
            ]

            if not files_needing_sync:
                logger.info("All files up to date, skipping metadata parsing")
                return

            logger.debug(
                "Parsing metadata only for files needing sync",
                total=len(self._model_index),
                need_sync=len(files_needing_sync),
            )
            gguf_files = [m.path for m in files_needing_sync]

        if gguf_files:
            with ParallelGGUFParser() as parser:
                metadata_map = parser.parse_files(gguf_files)

            for filename, info in self._model_index.items():
                if info.path in metadata_map:
                    info.metadata = metadata_map[info.path]

        logger.info("Built model index", count=len(self._model_index))

    def _build_groups(self) -> None:
        """Build model groups from index."""
        logger.debug("Building model groups")

        self._group_index = {}

        # Temporary storage for grouping
        multipart_files: dict[str, list[ModelInfo]] = defaultdict(list)
        mmproj_files: dict[str, ModelInfo] = {}
        single_files: dict[str, ModelInfo] = {}

        # First pass: categorize files
        for filename, info in self._model_index.items():
            if info.is_mmproj:
                base = get_mmproj_base(filename)
                if base:
                    mmproj_files[base] = info
                    logger.debug("Found mmproj file", filename=filename, base=base)
                else:
                    logger.debug("Could not extract base from mmproj filename", filename=filename)
                continue

            multipart_base = get_multipart_base(filename)
            if multipart_base:
                multipart_files[multipart_base].append(info)
            else:
                base = filename.replace(".gguf", "")
                single_files[base] = info

        # Create groups for multipart models
        for base_name, files in multipart_files.items():
            group = ModelGroup(
                base_name=base_name,
                files=sorted(files, key=lambda f: f.name),
                mmproj_file=mmproj_files.get(base_name),
                source_dir=self.source_dir,
            )
            self._group_index[group.model_id] = group

        # Create groups for single-file models
        for base_name, info in single_files.items():
            group = ModelGroup(
                base_name=base_name,
                files=[info],
                mmproj_file=mmproj_files.get(base_name),
                source_dir=self.source_dir,
            )
            self._group_index[group.model_id] = group

        # Try to match unmatched mmproj files
        for mmproj_base, mmproj_info in mmproj_files.items():
            matched = False
            for group in self._group_index.values():
                # Check if mmproj base matches group base
                group_base_stripped = strip_quantization_suffix(group.base_name)
                mmproj_base_stripped = strip_quantization_suffix(mmproj_base)

                logger.debug(
                    "Trying to match mmproj",
                    mmproj_base=mmproj_base,
                    mmproj_base_stripped=mmproj_base_stripped,
                    group_base=group.base_name,
                    group_base_stripped=group_base_stripped,
                )

                if (
                    group_base_stripped == mmproj_base_stripped
                    or group.base_name in mmproj_base
                    or mmproj_base in group.base_name
                ):
                    if group.mmproj_file is None:
                        group.mmproj_file = mmproj_info
                        matched = True
                        logger.info(
                            "Matched mmproj to group",
                            mmproj_file=mmproj_info.name,
                            group_id=group.model_id,
                        )
                        break

            if not matched:
                logger.warning("Unmatched mmproj file", file=mmproj_info.name, base=mmproj_base)

        logger.info("Built model groups", count=len(self._group_index))

    def _sync_to_backend(self, backend: Backend) -> BackendResult:
        """Sync all groups to a specific backend.

        Args:
            backend: Backend to sync to

        Returns:
            BackendResult with operation results
        """
        result = BackendResult(success=True)

        for group in self._group_index.values():
            # Check if this model should be skipped for this backend
            if self._should_skip_backend(backend, group.model_id):
                logger.info(
                    "Skipping model for backend (ignore pattern)",
                    model_id=group.model_id,
                    backend=backend.name,
                )
                result.skipped += 1
                result.skip_reasons.append(
                    {
                        "item": group.model_id,
                        "reason": f"ignored for {backend.name}",
                        "backend": backend.name,
                    }
                )
                continue

            # Get effective context size for this backend/model
            metadata = group.primary_file.metadata if group.primary_file else None
            context_size = self._get_context_size(backend, metadata)

            group_result = backend.sync_group(group, self.source_dir, context_size)
            result.linked += group_result.linked
            result.updated += group_result.updated
            result.skipped += group_result.skipped
            result.errors.extend(group_result.errors)
            result.skip_reasons.extend(group_result.skip_reasons)

        return result

    def _handle_deletion(self, event: SyncEvent) -> dict[str, BackendResult]:
        """Handle file deletion event.

        Args:
            event: Deletion event

        Returns:
            Dictionary of backend results
        """
        results = {}
        filename = event.path.name

        # Determine if this is from source or a backend
        is_source = event.source_dir == self.source_dir

        if is_source:
            # File deleted from source
            if self.config.sync.add_only:
                # Add-only mode: don't remove from backends, just log
                logger.info(
                    "File deleted from source (add-only mode, not removing from backends)",
                    file=filename,
                )
                return results

            logger.info("File deleted from source", file=filename)

            # Find which group this file belonged to
            affected_groups = []
            for group in self._group_index.values():
                if any(f.name == filename for f in group.files):
                    affected_groups.append(group)
                    break

            # If multipart file, check if whole group should be removed
            for group in affected_groups:
                # Check if any files remain for this group on disk
                remaining = sum(1 for f in group.files if f.path.exists())

                if remaining == 0:
                    # Group is gone, remove from all backends
                    for backend in self.backends:
                        result = backend.remove_group(group.model_id)
                        results[backend.name] = result

                    # Remove from our index
                    if group.model_id in self._group_index:
                        del self._group_index[group.model_id]

            # Remove from model index
            if filename in self._model_index:
                del self._model_index[filename]

        else:
            # File deleted from backend - restore from source (if not add-only)
            if self.config.sync.add_only:
                logger.info(
                    "File deleted from backend (add-only mode, not restoring)",
                    file=filename,
                )
                return results

            logger.info("File deleted from backend, will restore", file=filename)

            # Find file in source
            source_file = self.source_dir / filename
            if source_file.exists():
                # Re-sync the affected group
                for group in self._group_index.values():
                    if any(f.name == filename for f in group.files):
                        for backend in self.backends:
                            metadata = group.primary_file.metadata if group.primary_file else None
                            context_size = self._get_context_size(backend, metadata)
                            result = backend.sync_group(group, self.source_dir, context_size)
                            results[backend.name] = result
                        break
            else:
                logger.warning("Deleted file not in source, cannot restore", file=filename)

        return results

    def _handle_creation(self, event: SyncEvent) -> dict[str, BackendResult]:
        """Handle file creation/modification event.

        Also checks for existing mmproj files in the same directory
        that might have been missed during --no-initial-sync startup.

        Args:
            event: Creation event

        Returns:
            Dictionary of backend results
        """
        results = {}
        path = event.path

        # Parse metadata only if needed
        metadata = None
        if self._backends_need_metadata():
            try:
                metadata = parse_gguf_file(path)
            except Exception as e:
                logger.warning("Failed to parse metadata", path=str(path), error=str(e))

        # Create ModelInfo
        stat = path.stat()
        model_info = ModelInfo(
            path=path,
            metadata=metadata,
            file_size=stat.st_size,
            mtime=stat.st_mtime,
        )

        # Update index
        self._model_index[path.name] = model_info

        # Check for existing mmproj files in the same directory
        # (important when using --no-initial-sync where mmproj might already exist)
        self._check_for_existing_mmproj_files(path.parent)

        # Rebuild groups (simpler than incremental update for this case)
        self._build_groups()

        # Find the group this file belongs to
        for group in self._group_index.values():
            if any(f.name == path.name for f in group.files):
                # Sync this group to all backends
                for backend in self.backends:
                    metadata = group.primary_file.metadata if group.primary_file else None
                    context_size = self._get_context_size(backend, metadata)
                    result = backend.sync_group(group, self.source_dir, context_size)
                    results[backend.name] = result
                break

        return results

    def _check_for_existing_mmproj_files(self, directory: Path) -> None:
        """Check for existing mmproj files in a directory and add them to index.

        This is important when using --no-initial-sync, as mmproj files
        that already exist won't be detected by the file watcher.

        Args:
            directory: Directory to check for mmproj files
        """
        try:
            for file_path in directory.glob("*.gguf"):
                if file_path.name in self._model_index:
                    continue  # Already indexed

                # Check if it's an mmproj file
                info = ModelInfo(path=file_path)
                if info.is_mmproj:
                    try:
                        stat = file_path.stat()
                        info.file_size = stat.st_size
                        info.mtime = stat.st_mtime
                        self._model_index[file_path.name] = info
                        logger.debug(
                            "Found existing mmproj file",
                            filename=file_path.name,
                            directory=str(directory),
                        )
                    except OSError as e:
                        logger.warning(
                            "Failed to stat mmproj file",
                            path=str(file_path),
                            error=str(e),
                        )
        except OSError as e:
            logger.warning(
                "Failed to check directory for mmproj files",
                directory=str(directory),
                error=str(e),
            )

    def get_stats(self) -> dict[str, Any]:
        """Get sync engine statistics.

        Returns:
            Dictionary with statistics
        """
        return {
            "source_dir": str(self.source_dir),
            "total_files": len(self._model_index),
            "total_groups": len(self._group_index),
            "backends": [b.name for b in self.backends],
        }
