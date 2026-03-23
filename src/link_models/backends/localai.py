"""LocalAI backend implementation."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Any

import yaml

from .base import Backend, BackendResult, SyncAction
from ..core.logging import get_logger
from ..core.models import LocalAIConfig, ModelGroup, GGUFMetadata

logger = get_logger(__name__)


class LocalAIBackend(Backend):
    """Backend for LocalAI model organization."""
    
    def __init__(self, config: LocalAIConfig) -> None:
        """Initialize LocalAI backend.
        
        Args:
            config: LocalAI configuration
        """
        super().__init__(config)
        self.localai_config = config
    
    @property
    def name(self) -> str:
        """Return backend name."""
        return "LocalAI"
    
    def setup(self) -> None:
        """Setup LocalAI backend directories."""
        super().setup()
        
        if not self.config.enabled:
            return
        
        self.models_dir = self.output_dir
        self._ensure_dir(self.models_dir)
    
    def sync_group(self, group: ModelGroup, source_dir: Path, context_size: int | None = None) -> BackendResult:
        """Sync a model group to LocalAI backend.
        
        Creates hardlinks in subdirectories and generates YAML configs.
        
        Args:
            group: Model group to sync
            context_size: Optional context size override
            source_dir: Source directory (ground truth)
            
        Returns:
            BackendResult with operation results
        """
        if not self.config.enabled:
            return BackendResult(
                success=True,
                skipped=1,
                skip_reasons=[{"item": group.model_id, "reason": "backend disabled"}],
            )
        
        result = BackendResult(success=True)
        model_id = group.model_id
        
        # Determine subdirectory path (LocalAI can use flat or subdir)
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
                    result.skip_reasons.append({
                        "item": model_file.name,
                        "reason": "file not in source directory",
                    })
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
                    result.skip_reasons.append({
                        "item": model_file.name,
                        "reason": "file not in source directory",
                    })
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
                    result.skip_reasons.append({
                        "item": model_file.name,
                        "reason": "already up-to-date",
                    })
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
        
        # Generate/update YAML config
        if self.localai_config.generate_yaml:
            yaml_path = self._get_yaml_path(model_id)
            needs_update = self._yaml_needs_update(yaml_path, group)
            
            if needs_update:
                self._generate_yaml(group, yaml_path)
                result.details["yaml_generated"] = str(yaml_path)
        
        return result
    
    def remove_group(self, model_id: str) -> BackendResult:
        """Remove a model group from LocalAI backend.
        
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
        
        # Remove YAML config
        yaml_path = self._get_yaml_path(model_id)
        if yaml_path.exists():
            if self._remove_path(yaml_path):
                result.removed += 1
        
        return result
    
    def _get_yaml_path(self, model_id: str) -> Path:
        """Get the path for a model's YAML config file."""
        filename = f"{self.localai_config.yaml_prefix}{model_id}.yaml"
        return self.models_dir / filename
    
    def _yaml_needs_update(self, yaml_path: Path, group: ModelGroup) -> bool:
        """Check if YAML config needs to be regenerated.
        
        Args:
            yaml_path: Path to YAML file
            group: Model group
            
        Returns:
            True if YAML needs update
        """
        if not yaml_path.exists():
            return True
        
        yaml_mtime = yaml_path.stat().st_mtime
        
        # Check if any source file is newer
        for model_file in group.get_all_files():
            if model_file.path.stat().st_mtime > yaml_mtime:
                return True
        
        return False
    
    def _generate_yaml(self, group: ModelGroup, yaml_path: Path) -> None:
        """Generate LocalAI YAML config for a model group.
        
        Args:
            group: Model group
            yaml_path: Output YAML path
        """
        model_id = group.model_id
        primary = group.primary_file
        
        if not primary:
            logger.warning("No primary file for group", model_id=model_id)
            return
        
        # Get metadata from primary file
        metadata = primary.metadata or GGUFMetadata()
        
        # Build file list
        files_section = []
        for model_file in sorted(group.files, key=lambda f: f.name):
            files_section.append({"filename": f"{model_id}/{model_file.name}"})
        
        # Determine backend
        backend = metadata.get_backend()
        
        # Build config dict
        # Use display_name (from filename) instead of metadata.name
        # metadata.name can contain unreliable data from GGUF header
        config = {
            "name": group.display_name,
            "files": files_section,
            "parameters": {
                "model": f"{model_id}/{primary.name}",
            },
            "backend": backend,
            "mmap": self.localai_config.mmap,
            "f16": self.localai_config.f16,
        }
        
        # Add optional fields
        if metadata.context_length:
            config["context_size"] = metadata.context_length
        
        config["gpu_layers"] = self.localai_config.gpu_layers
        
        # Add mmproj if present
        if group.mmproj_file:
            config["mmproj"] = f"{model_id}/{group.mmproj_file.name}"
        
        # Add stop tokens if available
        if metadata.stop_tokens:
            config["stopwords"] = metadata.stop_tokens
        
        # Generate YAML with header comment
        yaml_content = self._yaml_with_header(config, primary.path)
        
        with open(yaml_path, "w", encoding="utf-8") as f:
            f.write(yaml_content)
        
        self._set_permissions(yaml_path)
        logger.info("Generated YAML config", path=str(yaml_path), model=model_id)
    
    def _yaml_with_header(self, config: dict, source_path: Path) -> str:
        """Generate YAML with header comment.
        
        Args:
            config: Configuration dict
            source_path: Source model path
            
        Returns:
            YAML string with header
        """
        header = f"""# Auto-generated LocalAI model configuration
# Generated: {datetime.now().isoformat()}
# Source: {source_path}
#
# Model: {config['name']}
# Backend: {config['backend']}

"""
        
        yaml_content = yaml.dump(
            config,
            default_flow_style=False,
            allow_unicode=True,
            sort_keys=False,
        )
        
        return header + yaml_content
    
    def cleanup_orphans(self, valid_model_ids: set[str]) -> BackendResult:
        """Remove orphaned model directories and YAMLs not in valid set.
        
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
                    result.errors.append(f"Failed to remove orphan dir: {item}")
        
        # Cleanup orphaned YAML files
        for yaml_file in self.models_dir.glob(f"{self.localai_config.yaml_prefix}*.yaml"):
            # Extract model_id from filename
            model_id = yaml_file.stem[len(self.localai_config.yaml_prefix):]
            
            if model_id not in valid_model_ids:
                if self._remove_path(yaml_file):
                    result.removed += 1
                else:
                    result.errors.append(f"Failed to remove orphan YAML: {yaml_file}")
        
        return result
