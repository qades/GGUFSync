"""Ollama backend implementation."""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any

from .base import Backend, BackendResult, SyncAction
from ..core.logging import get_logger
from ..core.models import OllamaConfig, ModelGroup, GGUFMetadata

logger = get_logger(__name__)


class OllamaBackend(Backend):
    """Backend for Ollama model organization.

    Ollama uses a manifest-based system with optional Modelfiles.
    This backend creates symlinks in the models directory and generates
    manifests for model discovery.
    """

    def __init__(self, config: OllamaConfig) -> None:
        """Initialize Ollama backend.

        Args:
            config: Ollama configuration
        """
        super().__init__(config)
        self.ollama_config = config

    @property
    def name(self) -> str:
        """Return backend name."""
        return "Ollama"

    def setup(self) -> None:
        """Setup Ollama backend directories."""
        super().setup()

        if not self.config.enabled:
            return

        self.models_dir = self.output_dir
        self._ensure_dir(self.models_dir)

        self.manifests_dir = self.models_dir / "manifests"
        self._ensure_dir(self.manifests_dir)

    def sync_group(self, group: ModelGroup, source_dir: Path, context_size: int | None = None) -> BackendResult:
        """Sync a model group to Ollama backend.

        Creates symlinks and generates manifest files if enabled.

        Args:
            group: Model group to sync
            context_size: Optional context size override
            source_dir: Source directory (ground truth)

        Returns:
            BackendResult with operation results
        """
        if not self.config.enabled:
            return BackendResult(success=True, skipped=1)

        result = BackendResult(success=True)
        model_id = group.model_id

        # Ollama uses flat structure with symlinked files
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

        # Generate Modelfile if enabled
        if self.ollama_config.generate_modelfile:
            self._generate_modelfile(group)

        return result

    def remove_group(self, model_id: str) -> BackendResult:
        """Remove a model group from Ollama backend.

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

        # Remove manifest file
        manifest_path = self._get_manifest_path(model_id)
        if manifest_path.exists():
            if self._remove_path(manifest_path):
                result.removed += 1

        return result

    def _get_manifest_path(self, model_id: str) -> Path:
        """Get the path for a model's manifest file."""
        return self.manifests_dir / f"{model_id}.json"

    def _generate_modelfile(self, group: ModelGroup) -> None:
        """Generate Ollama Modelfile for a model group.

        Args:
            group: Model group
        """
        model_id = group.model_id
        primary = group.primary_file

        if not primary:
            logger.warning("No primary file for group", model_id=model_id)
            return

        metadata = primary.metadata or GGUFMetadata()

        # Build Modelfile content
        modelfile_content = self._build_modelfile_content(group, metadata)

        modelfile_path = self.models_dir / model_id / "Modelfile"

        with open(modelfile_path, "w", encoding="utf-8") as f:
            f.write(modelfile_content)

        self._set_permissions(modelfile_path)
        logger.debug("Generated Ollama Modelfile", path=str(modelfile_path))

    def _build_modelfile_content(self, group: ModelGroup, metadata: GGUFMetadata) -> str:
        """Build Modelfile content.

        Args:
            group: Model group
            metadata: GGUF metadata

        Returns:
            Modelfile content string
        """
        lines = [
            f"# Auto-generated Ollama Modelfile",
            f"# Generated: {datetime.now().isoformat()}",
            f"# Model: {group.display_name}",
            "",
            f"FROM ./{group.primary_file.name}",
            "",
        ]

        # Add system prompt if available in additional_params
        if self.ollama_config.additional_params.get("system_prompt"):
            lines.append(f'SYSTEM "{self.ollama_config.additional_params["system_prompt"]}"')
            lines.append("")

        # Add parameters
        params = self.ollama_config.additional_params
        if params:
            for key, value in params.items():
                if key == "system_prompt":
                    continue
                if isinstance(value, bool):
                    if value:
                        lines.append(f"PARAMETER {key}")
                elif isinstance(value, str):
                    lines.append(f"PARAMETER {key} {value}")
                else:
                    lines.append(f"PARAMETER {key} {value}")
            lines.append("")

        # Add template if we have chat template metadata
        if metadata.chat_template:
            lines.append(f'TEMPLATE """{metadata.chat_template}"""')
            lines.append("")

        return "\n".join(lines)

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
            if item.name in ("manifests",):
                continue

            if item.is_dir() and item.name not in valid_model_ids:
                if self._remove_path(item):
                    result.removed += 1
                else:
                    result.errors.append(f"Failed to remove orphan: {item}")

        # Cleanup orphaned manifest files
        if self.manifests_dir.exists():
            for manifest_file in self.manifests_dir.glob("*.json"):
                model_id = manifest_file.stem

                if model_id not in valid_model_ids:
                    if self._remove_path(manifest_file):
                        result.removed += 1
                    else:
                        result.errors.append(f"Failed to remove orphan manifest: {manifest_file}")

        return result
