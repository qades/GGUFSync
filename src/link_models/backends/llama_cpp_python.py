"""llama-cpp-python backend implementation."""

from __future__ import annotations

from pathlib import Path

from .base import Backend, BackendResult
from ..core.logging import get_logger
from ..core.models import LlamaCppPythonConfig, ModelGroup

logger = get_logger(__name__)


class LlamaCppPythonBackend(Backend):
    """Backend for llama-cpp-python API server.

    llama-cpp-python is a Python binding for llama.cpp that runs as an API server.
    This backend creates symlinks to model files that the server can access.
    """

    def __init__(self, config: LlamaCppPythonConfig) -> None:
        super().__init__(config)
        self.lcpp_config = config

    @property
    def name(self) -> str:
        return "llama-cpp-python"

    def setup(self) -> None:
        super().setup()

        if not self.config.enabled:
            return

        self.models_dir = self.output_dir
        self._ensure_dir(self.models_dir)

    def sync_group(self, group: ModelGroup, source_dir: Path) -> BackendResult:
        if not self.config.enabled:
            return BackendResult(success=True, skipped=1)

        model_dir = self.models_dir / group.model_id
        self._ensure_dir(model_dir)

        return self._link_model_files(group, model_dir, source_dir)

    def remove_group(self, model_id: str) -> BackendResult:
        result = BackendResult(success=True)

        model_dir = self.models_dir / model_id
        if model_dir.exists():
            if self._remove_path(model_dir):
                result.removed += 1
            else:
                result.errors.append(f"Failed to remove {model_dir}")

        return result

    def cleanup_orphans(self, valid_model_ids: set[str]) -> BackendResult:
        return self._cleanup_orphans_simple(self.models_dir, valid_model_ids)
