"""GPT4All backend implementation."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

from .base import Backend, BackendResult, SyncAction
from ..core.logging import get_logger
from ..core.models import GPT4AllConfig, ModelGroup, GGUFMetadata

logger = get_logger(__name__)


class GPT4AllBackend(Backend):
    """Backend for GPT4All model organization.

    GPT4All uses a flat directory structure with optional config files.
    Models can be configured via the GUI or via JSON config.
    """

    def __init__(self, config: GPT4AllConfig) -> None:
        """Initialize GPT4All backend.

        Args:
            config: GPT4All configuration
        """
        super().__init__(config)
        self.gpt4all_config = config

    @property
    def name(self) -> str:
        """Return backend name."""
        return "GPT4All"

    def setup(self) -> None:
        """Setup GPT4All backend directories."""
        super().setup()

        if not self.config.enabled:
            return

        self.models_dir = self.output_dir
        self._ensure_dir(self.models_dir)

        if self.gpt4all_config.generate_config:
            self.configs_dir = self.models_dir / ".configs"
            self._ensure_dir(self.configs_dir)

    def sync_group(self, group: ModelGroup, source_dir: Path) -> BackendResult:
        """Sync a model group to GPT4All backend.

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

        # GPT4All uses flat directory structure
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

        # Generate config if enabled
        if self.gpt4all_config.generate_config:
            self._generate_config(group)

        return result

    def remove_group(self, model_id: str) -> BackendResult:
        """Remove a model group from GPT4All backend.

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

        # Remove config file if exists
        if hasattr(self, "configs_dir") and self.configs_dir:
            config_file = self.configs_dir / f"{model_id}.json"
            if config_file.exists():
                if self._remove_path(config_file):
                    result.removed += 1

        return result

    def _generate_config(self, group: ModelGroup) -> None:
        """Generate JSON config for GPT4All.

        Args:
            group: Model group
        """
        model_id = group.model_id
        primary = group.primary_file

        if not primary:
            logger.warning("No primary file for group", model_id=model_id)
            return

        metadata = primary.metadata or GGUFMetadata()

        # Build config - GPT4All uses a specific JSON format
        config = {
            "model": f"{model_id}/{primary.name}",
            "model_name": group.display_name,
            "model_path": str(primary.path),
            "context_length": metadata.context_length or self.gpt4all_config.default_context_size,
            "parameters": {
                "temperature": 0.7,
                "top_p": 0.9,
                "top_k": 40,
                "repeat_penalty": 1.1,
                "frequency_penalty": 0.0,
                "presence_penalty": 0.0,
            },
            "llm": {
                "gpu_layers": self.gpt4all_config.default_gpu_layers,
            },
        }

        if group.has_vision and group.mmproj_file:
            config["mmproj"] = f"{model_id}/{group.mmproj_file.name}"

        # Add chat template if available
        if metadata.chat_template:
            config["chat_template"] = metadata.chat_template

        config_path = self.configs_dir / f"{model_id}.json"

        import json

        with open(config_path, "w", encoding="utf-8") as f:
            json.dump(config, f, indent=2)

        self._set_permissions(config_path)
        logger.debug("Generated GPT4All config", path=str(config_path))

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
            if item.name in (".configs",):
                continue

            if item.is_dir() and item.name not in valid_model_ids:
                if self._remove_path(item):
                    result.removed += 1
                else:
                    result.errors.append(f"Failed to remove orphan: {item}")

        # Cleanup config files
        if hasattr(self, "configs_dir") and self.configs_dir and self.configs_dir.exists():
            for config_file in self.configs_dir.glob("*.json"):
                model_id = config_file.stem

                if model_id not in valid_model_ids:
                    if self._remove_path(config_file):
                        result.removed += 1

        return result
