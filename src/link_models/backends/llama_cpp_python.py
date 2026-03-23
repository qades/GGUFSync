"""llama-cpp-python backend implementation."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

from .base import Backend, BackendResult, SyncAction
from ..core.logging import get_logger
from ..core.models import LlamaCppPythonConfig, ModelGroup, GGUFMetadata

logger = get_logger(__name__)


class LlamaCppPythonBackend(Backend):
    """Backend for llama-cpp-python API server.

    llama-cpp-python is a Python binding for llama.cpp that runs as an API server.
    This backend creates symlinks to model files that the server can access.
    """

    def __init__(self, config: LlamaCppPythonConfig) -> None:
        """Initialize llama-cpp-python backend.

        Args:
            config: llama-cpp-python configuration
        """
        super().__init__(config)
        self.lcpp_config = config

    @property
    def name(self) -> str:
        """Return backend name."""
        return "llama-cpp-python"

    def setup(self) -> None:
        """Setup llama-cpp-python backend directories."""
        super().setup()

        if not self.config.enabled:
            return

        self.models_dir = self.output_dir
        self._ensure_dir(self.models_dir)

    def sync_group(self, group: ModelGroup, source_dir: Path) -> BackendResult:
        """Sync a model group to llama-cpp-python backend.

        Creates symlinks in the models directory.

        Args:
            group: Model group to sync
            source_dir: Source directory (ground truth)

        Returns:
            BackendResult with operation results
        """
        if not self.config.enabled:
            return BackendResult(success=True, skipped=1)

        result = BackendResult(success=True)
        model_id = group.model_id

        model_dir = self.models_dir / model_id
        self._ensure_dir(model_dir)

        for model_file in group.files:
            if not model_file.path.is_relative_to(source_dir):
                logger.warning(
                    "File not in source directory, skipping",
                    file=model_file.name,
                    source=str(source_dir),
                )
                result.skipped += 1
                continue

            target = model_dir / model_file.name
            link_result = self._create_link(
                model_file.path,
                target,
                prefer_hardlink=self.config.prefer_hardlinks,
            )

            if link_result.success:
                if link_result.action == SyncAction.CREATE:
                    result.linked += 1
                elif link_result.action == SyncAction.SKIP:
                    result.skipped += 1
                elif link_result.action == SyncAction.UPDATE:
                    result.updated += 1
            else:
                result.errors.append(link_result.error or "Unknown error")

        if group.mmproj_file:
            mmproj_target = model_dir / group.mmproj_file.name
            link_result = self._create_link(
                group.mmproj_file.path,
                mmproj_target,
                prefer_hardlink=self.config.prefer_hardlinks,
            )

            if link_result.success:
                if link_result.action == SyncAction.CREATE:
                    result.linked += 1

        return result

    def remove_group(self, model_id: str) -> BackendResult:
        """Remove a model group from llama-cpp-python backend.

        Args:
            model_id: Normalized model ID to remove

        Returns:
            BackendResult with operation results
        """
        result = BackendResult(success=True)

        model_dir = self.models_dir / model_id
        if model_dir.exists():
            if self._remove_path(model_dir):
                result.removed += 1
            else:
                result.errors.append(f"Failed to remove {model_dir}")

        return result

    def cleanup_orphans(self, valid_model_ids: set[str]) -> BackendResult:
        """Remove orphaned model directories not in valid set.

        Args:
            valid_model_ids: Set of valid model IDs

        Returns:
            BackendResult with cleanup results
        """
        result = BackendResult(success=True)

        if not self.models_dir.exists():
            return result

        for item in self.models_dir.iterdir():
            if not item.is_dir():
                continue

            if item.name not in valid_model_ids:
                if self._remove_path(item):
                    result.removed += 1
                else:
                    result.errors.append(f"Failed to remove orphan: {item}")

        return result
