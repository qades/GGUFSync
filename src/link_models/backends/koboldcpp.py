"""KoboldCpp backend implementation."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .base import Backend, BackendResult, SyncAction
from ..core.logging import get_logger
from ..core.models import KoboldCppConfig, ModelGroup, GGUFMetadata

logger = get_logger(__name__)


class KoboldCppBackend(Backend):
    """Backend for KoboldCpp model organization.

    KoboldCpp uses .kcpps sidecar config files alongside model files.
    This backend creates symlinks and generates .kcpps files.
    """

    def __init__(self, config: KoboldCppConfig) -> None:
        """Initialize KoboldCpp backend.

        Args:
            config: KoboldCpp configuration
        """
        super().__init__(config)
        self.koboldcpp_config = config

    @property
    def name(self) -> str:
        """Return backend name."""
        return "KoboldCpp"

    def setup(self) -> None:
        """Setup KoboldCpp backend directories."""
        super().setup()

        if not self.config.enabled:
            return

        self.models_dir = self.output_dir
        self._ensure_dir(self.models_dir)

    def sync_group(self, group: ModelGroup, source_dir: Path) -> BackendResult:
        """Sync a model group to KoboldCpp backend.

        Creates symlinks and generates .kcpps sidecar files.

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

        # KoboldCpp uses flat directory structure
        model_subdir = self.models_dir / model_id
        self._ensure_dir(model_subdir)

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

            target = model_subdir / model_file.name
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
            mmproj_target = model_subdir / group.mmproj_file.name
            link_result = self._create_link(
                group.mmproj_file.path,
                mmproj_target,
                prefer_hardlink=self.config.prefer_hardlinks,
            )

            if link_result.success:
                if link_result.action == SyncAction.CREATE:
                    result.linked += 1

        # Generate .kcpps file if enabled
        if self.koboldcpp_config.generate_kcpps:
            self._generate_kcpps(group)

        return result

    def remove_group(self, model_id: str) -> BackendResult:
        """Remove a model group from KoboldCpp backend.

        Args:
            model_id: Normalized model ID to remove

        Returns:
            BackendResult with operation results
        """
        result = BackendResult(success=True)

        # Remove model directory
        model_subdir = self.models_dir / model_id
        if model_subdir.exists():
            if self._remove_path(model_subdir):
                result.removed += 1
            else:
                result.errors.append(f"Failed to remove {model_subdir}")

        return result

    def _generate_kcpps(self, group: ModelGroup) -> None:
        """Generate .kcpps sidecar config for KoboldCpp.

        Args:
            group: Model group
        """
        model_id = group.model_id
        primary = group.primary_file

        if not primary:
            logger.warning("No primary file for group", model_id=model_id)
            return

        metadata = primary.metadata or GGUFMetadata()

        # Build .kcpps config (KoboldCpp uses JSON-like format saved as .kcpps)
        config = {
            "model_param": f"./{primary.name}",
            "contextsize": metadata.context_length or self.koboldcpp_config.default_context_size,
            "gpulayers": self.koboldcpp_config.default_gpu_layers,
            "threads": self.koboldcpp_config.default_threads,
            "use_mmap": True,
            "use_flash_attention": True,
        }

        # Add mmproj if present
        if group.mmproj_file:
            config["mmproj_param"] = f"./{group.mmproj_file.name}"

        # Add stopwords if available
        if metadata.stop_tokens:
            config["stop"] = metadata.stop_tokens

        kcpps_path = self.models_dir / f"{model_id}.kcpps"

        with open(kcpps_path, "w", encoding="utf-8") as f:
            json.dump(config, f, indent=2)

        self._set_permissions(kcpps_path)
        logger.debug("Generated .kcpps config", path=str(kcpps_path))

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

        # Cleanup .kcpps files
        for kcpps_file in self.models_dir.glob("*.kcpps"):
            model_id = kcpps_file.stem

            if model_id not in valid_model_ids:
                if self._remove_path(kcpps_file):
                    result.removed += 1
                else:
                    result.errors.append(f"Failed to remove orphan .kcpps: {kcpps_file}")

        return result
