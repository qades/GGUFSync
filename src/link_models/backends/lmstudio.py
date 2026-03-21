"""LM Studio backend implementation."""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any

from .base import Backend, BackendResult, SyncAction
from ..core.logging import get_logger
from ..core.models import LMStudioConfig, ModelGroup, GGUFMetadata

logger = get_logger(__name__)


class LMStudioBackend(Backend):
    """Backend for LM Studio model organization.
    
    LM Studio uses a specific directory structure and manifest files
    to organize models for its UI.
    """
    
    def __init__(self, config: LMStudioConfig) -> None:
        """Initialize LM Studio backend.
        
        Args:
            config: LM Studio configuration
        """
        super().__init__(config)
        self.lmstudio_config = config
    
    @property
    def name(self) -> str:
        """Return backend name."""
        return "LM Studio"
    
    def setup(self) -> None:
        """Setup LM Studio backend directories."""
        super().setup()
        
        if not self.config.enabled:
            return
        
        self.models_dir = self.output_dir
        self._ensure_dir(self.models_dir)
        
        # LM Studio might use a manifest directory
        self.manifest_dir = self.models_dir / ".manifests"
        self._ensure_dir(self.manifest_dir)
    
    def sync_group(self, group: ModelGroup, source_dir: Path) -> BackendResult:
        """Sync a model group to LM Studio backend.
        
        LM Studio organizes models in a flat structure with metadata
        stored in sidecar files or a manifest.
        
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
        
        # LM Studio typically uses a flat structure but with subdirs for organization
        # We'll create a subdirectory per model like other backends
        model_subdir = self.models_dir / model_id
        self._ensure_dir(model_subdir)
        
        # Link all model files
        for model_file in group.files:
            # Source must be in source_dir
            try:
                if not model_file.path.is_relative_to(source_dir):
                    logger.warning(
                        "File not in source directory, skipping",
                        file=model_file.name,
                        source=str(source_dir),
                    )
                    result.skipped += 1
                    continue
            except AttributeError:
                # Python < 3.9 fallback
                try:
                    model_file.path.relative_to(source_dir)
                except ValueError:
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
        
        # Generate metadata file if enabled
        if self.lmstudio_config.generate_manifest:
            self._generate_manifest(group)
        
        return result
    
    def remove_group(self, model_id: str) -> BackendResult:
        """Remove a model group from LM Studio backend.
        
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
        return self.manifest_dir / f"{model_id}.json"
    
    def _generate_manifest(self, group: ModelGroup) -> None:
        """Generate LM Studio manifest for a model group.
        
        Args:
            group: Model group
        """
        model_id = group.model_id
        primary = group.primary_file
        
        if not primary:
            logger.warning("No primary file for group", model_id=model_id)
            return
        
        # Get metadata
        metadata = primary.metadata or GGUFMetadata()
        
        # Build manifest
        manifest = {
            "id": model_id,
            "name": metadata.name or model_id,
            "architecture": metadata.architecture,
            "files": [
                {
                    "name": f.name,
                    "size": f.file_size if hasattr(f, 'file_size') else f.path.stat().st_size,
                }
                for f in group.get_all_files()
            ],
            "quantization": metadata.quantization,
            "context_length": metadata.context_length,
            "parameters": {
                "vocab_size": metadata.vocab_size,
                "num_layers": metadata.num_hidden_layers,
            },
            "has_vision": group.has_vision,
            "modified_at": datetime.now().isoformat(),
        }
        
        manifest_path = self._get_manifest_path(model_id)
        
        with open(manifest_path, "w", encoding="utf-8") as f:
            json.dump(manifest, f, indent=2)
        
        self._set_permissions(manifest_path)
        logger.debug("Generated LM Studio manifest", path=str(manifest_path))
    
    def cleanup_orphans(self, valid_model_ids: set[str]) -> BackendResult:
        """Remove orphaned model directories and manifests not in valid set.
        
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
            if item.name.startswith("."):
                continue  # Skip hidden directories like .manifests
            
            if item.is_dir() and item.name not in valid_model_ids:
                if self._remove_path(item):
                    result.removed += 1
                else:
                    result.errors.append(f"Failed to remove orphan: {item}")
        
        # Cleanup orphaned manifest files
        if self.manifest_dir.exists():
            for manifest_file in self.manifest_dir.glob("*.json"):
                model_id = manifest_file.stem
                
                if model_id not in valid_model_ids:
                    if self._remove_path(manifest_file):
                        result.removed += 1
                    else:
                        result.errors.append(f"Failed to remove orphan manifest: {manifest_file}")
        
        return result
