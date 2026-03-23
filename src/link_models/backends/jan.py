"""Jan backend implementation."""

from __future__ import annotations

import json
from pathlib import Path

from .base import Backend, BackendResult
from ..core.logging import get_logger
from ..core.models import JanConfig, ModelGroup, GGUFMetadata

logger = get_logger(__name__)


class JanBackend(Backend):
    """Backend for Jan model organization.

    Jan (jan.ai) uses a flat directory structure with model metadata.
    Models are stored in the models/ subdirectory with config.json files.
    """

    def __init__(self, config: JanConfig) -> None:
        super().__init__(config)
        self.jan_config = config

    @property
    def name(self) -> str:
        return "Jan"

    def setup(self) -> None:
        super().setup()

        if not self.config.enabled:
            return

        self.models_dir = self.output_dir / "models"
        self._ensure_dir(self.models_dir)

    def sync_group(self, group: ModelGroup, source_dir: Path) -> BackendResult:
        if not self.config.enabled:
            return BackendResult(success=True, skipped=1)

        model_dir = self.models_dir / group.model_id
        self._ensure_dir(model_dir)

        result = self._link_model_files(group, model_dir, source_dir)

        if self.jan_config.generate_metadata:
            self._generate_metadata(group, model_dir)

        return result

    def remove_group(self, model_id: str) -> BackendResult:
        result = BackendResult(success=True)

        model_dir = self.models_dir / model_id
        if model_dir.exists():
            if self._remove_path(model_dir):
                result.removed += 1
            else:
                result.errors.append(f"Failed to remove {model_dir}")

        return result

    def _generate_metadata(self, group: ModelGroup, model_dir: Path) -> None:
        model_id = group.model_id
        primary = group.primary_file

        if not primary:
            logger.warning("No primary file for group", model_id=model_id)
            return

        metadata = primary.metadata or GGUFMetadata()

        config = {
            "id": model_id,
            "object": "model",
            "created": 0,
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
        return self._cleanup_orphans_simple(self.models_dir, valid_model_ids)
