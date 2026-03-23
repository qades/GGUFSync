"""Jan backend implementation."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .base import Backend, BackendResult, SyncAction
from ..core.logging import get_logger
from ..core.models import ModelGroup, GGUFMetadata

logger = get_logger(__name__)


class JanConfig:
    """Configuration for Jan backend."""

    def __init__(
        self,
        output_dir: Path,
        enabled: bool = True,
        generate_metadata: bool = True,
    ) -> None:
        self.output_dir = output_dir
        self.enabled = enabled
        self.generate_metadata = generate_metadata


class JanBackend(Backend):
    """Backend for Jan model organization.

    Jan (jan.ai) uses a flat directory structure with model metadata.
    Models are stored in the models/ subdirectory with config.json files.
    """

    def __init__(self, config: JanConfig) -> None:
        """Initialize Jan backend.

        Args:
            config: Jan configuration
        """
        super().__init__(config)
        self.jan_config = config

    @property
    def name(self) -> str:
        """Return backend name."""
        return "Jan"

    def setup(self) -> None:
        """Setup Jan backend directories."""
        super().setup()

        if not self.config.enabled:
            return

        self.models_dir = self.output_dir / "models"
        self._ensure_dir(self.models_dir)

    def sync_group(self, group: ModelGroup, source_dir: Path) -> BackendResult:
        """Sync a model group to Jan backend.

        Jan stores models flat with metadata in config.json.

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

        # Jan uses flat directory structure
        model_dir = self.models_dir / model_id
        self._ensure_dir(model_dir)

        # Link all model files
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

        # Link mmproj if present
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

        # Generate metadata file if enabled
        if self.jan_config.generate_metadata:
            self._generate_metadata(group, model_dir)

        return result

    def remove_group(self, model_id: str) -> BackendResult:
        """Remove a model group from Jan backend.

        Args:
            model_id: Normalized model ID to remove

        Returns:
            BackendResult with operation results
        """
        result = BackendResult(success=True)

        # Remove model directory
        model_dir = self.models_dir / model_id
        if model_dir.exists():
            if self._remove_path(model_dir):
                result.removed += 1
            else:
                result.errors.append(f"Failed to remove {model_dir}")

        return result

    def _generate_metadata(self, group: ModelGroup, model_dir: Path) -> None:
        """Generate Jan metadata file.

        Args:
            group: Model group
            model_dir: Directory to write metadata to
        """
        model_id = group.model_id
        primary = group.primary_file

        if not primary:
            logger.warning("No primary file for group", model_id=model_id)
            return

        metadata = primary.metadata or GGUFMetadata()

        # Build Jan config (model.json format)
        config = {
            "id": model_id,
            "object": "model",
            "created": 0,  # Will be set by Jan
            "owned_by": "user",
            "filename": primary.name,
            "size": primary.file_size if hasattr(primary, "file_size") else 0,
            "metadata": {
                "arch": metadata.architecture,
                "quantization": metadata.quantization,
                "context_length": metadata.context_length,
            },
        }

        config_path = model_dir / "model.json"

        with open(config_path, "w", encoding="utf-8") as f:
            json.dump(config, f, indent=2)

        self._set_permissions(config_path)
        logger.debug("Generated Jan metadata", path=str(config_path))

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

        # Cleanup model directories
        for item in self.models_dir.iterdir():
            if not item.is_dir():
                continue

            if item.name not in valid_model_ids:
                if self._remove_path(item):
                    result.removed += 1
                else:
                    result.errors.append(f"Failed to remove orphan: {item}")

        return result
