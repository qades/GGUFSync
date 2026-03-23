"""vLLM backend implementation."""

from __future__ import annotations

import json
from pathlib import Path

from .base import Backend, BackendResult, SyncAction
from ..core.logging import get_logger
from ..core.models import vLLMConfig, ModelGroup, GGUFMetadata

logger = get_logger(__name__)


class vLLMBackend(Backend):
    """Backend for vLLM model organization.

    vLLM uses HuggingFace-style directory structure with config.yaml.
    This backend creates symlinks and generates config files.
    """

    def __init__(self, config: vLLMConfig) -> None:
        """Initialize vLLM backend.

        Args:
            config: vLLM configuration
        """
        super().__init__(config)
        self.vllm_config = config

    @property
    def name(self) -> str:
        """Return backend name."""
        return "vLLM"

    def setup(self) -> None:
        """Setup vLLM backend directories."""
        super().setup()

        if not self.config.enabled:
            return

        self.models_dir = self.output_dir
        self._ensure_dir(self.models_dir)

        if self.vllm_config.generate_config:
            self.configs_dir = self.models_dir / ".configs"
            self._ensure_dir(self.configs_dir)

    def sync_group(self, group: ModelGroup, source_dir: Path) -> BackendResult:
        """Sync a model group to vLLM backend.

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

        # vLLM uses HuggingFace-style directory structure
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
        if self.vllm_config.generate_config:
            self._generate_config(group, model_subdir)

        return result

    def remove_group(self, model_id: str) -> BackendResult:
        """Remove a model group from vLLM backend.

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

    def _generate_config(self, group: ModelGroup, model_subdir: Path) -> None:
        """Generate vLLM config.yaml for a model group.

        Args:
            group: Model group
            model_subdir: Directory to write config to
        """
        model_id = group.model_id
        primary = group.primary_file

        if not primary:
            logger.warning("No primary file for group", model_id=model_id)
            return

        metadata = primary.metadata or GGUFMetadata()

        # Build config (vLLM uses HuggingFace-style config.json)
        config = {
            "model_type": metadata.architecture or "llama",
            "torch_dtype": "float16",
            "trust_remote_code": self.vllm_config.trust_remote_code,
        }

        # Add context length if known
        if metadata.context_length:
            config["max_model_len"] = metadata.context_length

        # Add quantization if known
        if metadata.quantization:
            config["quantization"] = f"gguf_{metadata.quantization}"

        # For vision models
        if group.has_vision and group.mmproj_file:
            config["mm_processor_kwargs"] = {
                "mm_model": f"./{group.mmproj_file.name}",
            }

        config_path = model_subdir / "config.json"

        import json

        with open(config_path, "w", encoding="utf-8") as f:
            json.dump(config, f, indent=2)

        self._set_permissions(config_path)
        logger.debug("Generated vLLM config.json", path=str(config_path))

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

        return result
