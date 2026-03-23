"""Text Generation WebUI (oobabooga) backend implementation."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import yaml

from .base import Backend, BackendResult, SyncAction
from ..core.logging import get_logger
from ..core.models import TextGenConfig, ModelGroup, GGUFMetadata

logger = get_logger(__name__)


class TextGenBackend(Backend):
    """Backend for Text Generation WebUI (oobabooga) model organization.

    TextGen uses a flat directory structure in user_data/models.
    It can optionally generate settings.yaml and per-model configs.
    """

    def __init__(self, config: TextGenConfig) -> None:
        """Initialize TextGen backend.

        Args:
            config: TextGen configuration
        """
        super().__init__(config)
        self.textgen_config = config

    @property
    def name(self) -> str:
        """Return backend name."""
        return "TextGen WebUI"

    def setup(self) -> None:
        """Setup TextGen backend directories."""
        super().setup()

        if not self.config.enabled:
            return

        self.models_dir = self.output_dir
        self._ensure_dir(self.models_dir)

        if self.textgen_config.generate_model_configs:
            self.configs_dir = self.models_dir / "config.yaml.d"
            self._ensure_dir(self.configs_dir)

    def sync_group(
        self,
        group: ModelGroup,
        source_dir: Path,
        context_size: int | None = None,
    ) -> BackendResult:
        """Sync a model group to TextGen backend.

        Creates symlinks and generates model configs.

        Args:
            group: Model group to sync
            source_dir: Source directory (ground truth)
            context_size: Optional context size override
            gpu_layers: Optional GPU layers override
            threads: Optional threads override
        Returns:
            BackendResult with operation results
        """
        """Sync a model group to TextGen backend.

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

        # TextGen uses flat directory structure
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

        # Generate model config if enabled
        if self.textgen_config.generate_model_configs:
            self._generate_model_config(group, context_size)

        return result

    def remove_group(self, model_id: str) -> BackendResult:
        """Remove a model group from TextGen backend.

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
            config_file = self.configs_dir / f"{model_id}.yaml"
            if config_file.exists():
                if self._remove_path(config_file):
                    result.removed += 1

        return result

    def _generate_model_config(self, group: ModelGroup, context_size: int | None = None) -> None:
        """Generate per-model config for TextGen.

        Args:
            group: Model group
            context_size: Optional context size override
        """
        model_id = group.model_id
        primary = group.primary_file

        if not primary:
            logger.warning("No primary file for group", model_id=model_id)
            return

        metadata = primary.metadata or GGUFMetadata()
        config_path = self.configs_dir / f"{model_id}.yaml"

        # Load existing config to preserve user settings
        existing = self._load_existing_config(config_path, "yaml")
        self.logger.debug(
            "TextGen config resolution",
            model=model_id,
            existing=existing is not None,
            context_size=context_size,
            config_context_size=self.config.context_size,
        )

        # Get context size: param > config > metadata > -1 (unlimited)
        effective_context_size = (
            context_size or self.config.context_size or metadata.context_length or -1
        )

        # Build default config
        defaults = {
            "model_name": group.display_name,
            "model_path": str(primary.path),
            "llm_loader": "llama.cpp",
            "settings": {
                "max_tokens": 512,
                "temperature": 0.7,
                "context_length": effective_context_size,
                "gpu_layers": -1,
            },
        }

        if group.has_vision and not existing:
            defaults["mmproj"] = str(group.mmproj_file.path) if group.mmproj_file else None

        # Merge with existing, preserving user values
        protected = {"model_name", "model_path", "llm_loader", "settings", "mmproj"}
        config = self._merge_config(existing, defaults, protected)

        with open(config_path, "w", encoding="utf-8") as f:
            yaml.dump(config, f, default_flow_style=False, sort_keys=False)

        self._set_permissions(config_path)
        logger.debug("Generated TextGen config", path=str(config_path))

    def generate_settings_yaml(self, output_path: Path | None = None) -> Path:
        """Generate global settings.yaml for TextGen.

        Args:
            output_path: Optional custom output path

        Returns:
            Path to generated settings file
        """
        settings_path = output_path or self.models_dir.parent / "settings.yaml"

        settings = {
            "default_extensions": [],
            "chat_default_extensions": ["gallery"],
            "presets": {
                "default": "Default",
                ".*(alpaca|llama|llava)": "LLaMA-Precise",
            },
            "truncation_length": 2048,
            "chat_prompt_size": 2048,
        }

        with open(settings_path, "w", encoding="utf-8") as f:
            yaml.dump(settings, f, default_flow_style=False, sort_keys=False)

        self._set_permissions(settings_path)
        logger.info("Generated TextGen settings.yaml", path=str(settings_path))
        return settings_path

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
            if item.name in ("config.yaml.d",):
                continue

            if item.is_dir() and item.name not in valid_model_ids:
                if self._remove_path(item):
                    result.removed += 1
                else:
                    result.errors.append(f"Failed to remove orphan: {item}")

        # Cleanup config files
        if hasattr(self, "configs_dir") and self.configs_dir and self.configs_dir.exists():
            for config_file in self.configs_dir.glob("*.yaml"):
                model_id = config_file.stem

                if model_id not in valid_model_ids:
                    if self._remove_path(config_file):
                        result.removed += 1

        return result
