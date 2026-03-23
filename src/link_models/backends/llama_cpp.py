"""llama.cpp backend implementation."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Any

from .base import Backend, BackendResult, LinkResult, SyncAction
from ..core.logging import get_logger, log_action
from ..core.models import LlamaCppConfig, ModelGroup

logger = get_logger(__name__)


class LlamaCppBackend(Backend):
    """Backend for llama.cpp model organization."""
    
    def __init__(self, config: LlamaCppConfig) -> None:
        """Initialize llama.cpp backend.
        
        Args:
            config: LlamaCpp configuration
        """
        super().__init__(config)
        self.llama_config = config
        self._group_map: dict[str, ModelGroup] = {}
    
    @property
    def name(self) -> str:
        """Return backend name."""
        return "llama.cpp"
    
    def setup(self) -> None:
        """Setup llama.cpp backend directories."""
        super().setup()
        
        if not self.config.enabled:
            return
        
        # Create subdirectory structure
        self.models_dir = self.output_dir
        self._ensure_dir(self.models_dir)
    
    def sync_group(self, group: ModelGroup, source_dir: Path, context_size: int | None = None) -> BackendResult:
        """Sync a model group to llama.cpp backend.
        
        Creates hardlinks in a subdirectory per model.
        
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
        
        # Determine subdirectory path
        model_subdir = self.models_dir / model_id
        self._ensure_dir(model_subdir)
        
        # Link all model files
        for model_file in group.files:
            # Source must be in source_dir (use try-except for Python < 3.9 compatibility)
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
                elif link_result.action == SyncAction.SKIP:
                    result.skipped += 1
        
        # Store group for models.ini generation
        self._group_map[model_id] = group
        
        # Generate models.ini if enabled
        if self.llama_config.generate_models_ini:
            self._update_models_ini()
        
        return result
    
    def remove_group(self, model_id: str) -> BackendResult:
        """Remove a model group from llama.cpp backend.
        
        Args:
            model_id: Normalized model ID to remove
            
        Returns:
            BackendResult with operation results
        """
        result = BackendResult(success=True)
        
        model_subdir = self.models_dir / model_id
        if model_subdir.exists():
            if self._remove_path(model_subdir):
                result.removed += 1
            else:
                result.errors.append(f"Failed to remove {model_subdir}")
        
        # Remove from group map
        self._group_map.pop(model_id, None)
        
        # Update models.ini
        if self.llama_config.generate_models_ini:
            self._update_models_ini()
        
        return result
    
    def _update_models_ini(self) -> None:
        """Generate or update models.ini file.
        
        Uses a custom format that is compatible with llama.cpp server:
        - Uses # for comments instead of [metadata] section
        - Section names use display_name (preserving capitalization)
        - Includes alias field for alternative access names
        """
        ini_path = (
            self.llama_config.models_ini_path
            or self.models_dir / "models.ini"
        )
        
        lines = []
        
        # Add header comments
        lines.append(f"# Auto-generated models.ini for llama.cpp")
        lines.append(f"# Generated: {datetime.now().isoformat()}")
        lines.append("")
        
        # Use group_map to get proper display names, or scan directories as fallback
        model_entries = []
        
        # First, collect info from known groups (for display names)
        for model_id, group in self._group_map.items():
            model_dir = self.models_dir / model_id
            if not model_dir.exists():
                continue
            
            # Find GGUF files
            gguf_files = list(model_dir.glob("*.gguf"))
            if not gguf_files:
                continue
            
            # Filter out mmproj files
            primary_files = [f for f in gguf_files if "mmproj" not in f.name.lower()]
            mmproj_files = [f for f in gguf_files if "mmproj" in f.name.lower()]
            
            if not primary_files:
                continue
            
            primary = sorted(primary_files)[0]
            
            # Use display_name from group (preserves original capitalization)
            display_name = group.display_name
            
            model_entries.append({
                "model_id": model_id,
                "display_name": display_name,
                "model": str(primary),
                "mmproj": str(sorted(mmproj_files)[0]) if mmproj_files else None,
            })
        
        # Check for any directories not in group_map (fallback)
        if self.models_dir.exists():
            for model_dir in sorted(self.models_dir.iterdir()):
                if not model_dir.is_dir():
                    continue
                
                model_id = model_dir.name
                # Skip if already processed from group_map
                if any(e["model_id"] == model_id for e in model_entries):
                    continue
                
                # Find GGUF files
                gguf_files = list(model_dir.glob("*.gguf"))
                if not gguf_files:
                    continue
                
                primary_files = [f for f in gguf_files if "mmproj" not in f.name.lower()]
                mmproj_files = [f for f in gguf_files if "mmproj" in f.name.lower()]
                
                if not primary_files:
                    continue
                
                primary = sorted(primary_files)[0]
                
                # Generate display name from directory name
                display_name = self._format_display_name(model_id)
                
                model_entries.append({
                    "model_id": model_id,
                    "display_name": display_name,
                    "model": str(primary),
                    "mmproj": str(sorted(mmproj_files)[0]) if mmproj_files else None,
                })
        
        # Write sections
        for entry in sorted(model_entries, key=lambda x: x["display_name"]):
            lines.append(f"[{entry['display_name']}]")
            lines.append(f"model = {entry['model']}")
            if entry['mmproj']:
                lines.append(f"mmproj = {entry['mmproj']}")
            
            # Add alias only if meaningfully different from display_name
            # (llama-server treats similar names as conflicts)
            # We compare normalized versions: lowercase, hyphens only
            if self._should_include_alias(entry['model_id'], entry['display_name']):
                lines.append(f"alias = {entry['model_id']}")
            
            lines.append("")
        
        # Write the ini file
        with open(ini_path, "w", encoding="utf-8") as f:
            f.write("\n".join(lines))
        
        self._set_permissions(ini_path)
        logger.info("Generated models.ini", path=str(ini_path), count=len(model_entries))
    
    def _should_include_alias(self, model_id: str, display_name: str) -> bool:
        """Check if alias should be included based on display_name difference.
        
        Llama-server treats names as conflicting if they're similar after
        normalization. We skip alias if it's just a case/separator difference.
        
        Llama-server's normalization (from server.cpp):
        - Converts to lowercase
        - Replaces underscores with dashes
        - Replaces dots with dashes (1.5b -> 1-5b)
        
        Args:
            model_id: Normalized model ID (e.g., 'arch-function-3b-q4-k-s')
            display_name: Formatted display name (e.g., 'Arch_Function_3B_Q4_K_S')
            
        Returns:
            True if alias should be included
        """
        logger.debug(
            "Checking if alias should be included",
            model_id=model_id,
            display_name=display_name,
        )
        
        if model_id == display_name:
            logger.debug("Skipping alias (exact match)")
            return False
        
        # Normalize display_name the same way llama-server does
        # This must match the normalization in server.cpp
        normalized_display = (
            display_name.lower()
            .replace("_", "-")
            .replace(".", "-")  # llama-server also replaces dots with dashes
        )
        
        logger.debug(
            "Normalized comparison",
            model_id=model_id,
            normalized_display=normalized_display,
        )
        
        # If they're the same after normalization, skip the alias
        # (llama-server would treat them as conflicts)
        if model_id == normalized_display:
            logger.debug(
                "Skipping alias (would conflict)",
                model_id=model_id,
                display_name=display_name,
            )
            return False
        
        return True
    
    def _format_display_name(self, model_id: str) -> str:
        """Format a model ID as a display name with proper capitalization.
        
        Converts 'llama-2-7b-chat' to 'Llama 2 7B Chat' etc.
        Preserves known acronyms like GGUF, Q4_K_M, etc.
        Replaces spaces with underscores for INI section compatibility.
        
        Args:
            model_id: Normalized model ID (lowercase with hyphens)
            
        Returns:
            Formatted display name safe for INI sections
        """
        # Split by hyphens
        parts = model_id.split("-")
        
        # Known acronyms to keep uppercase
        acronyms = {
            "gguf", "ggml", "gpt", "llm", "ai", "api", "cpu", "gpu", "ram", "vram",
            "q4", "q5", "q6", "q8", "f16", "f32", "bf16", "q2", "q3", "q4_k", "q5_k",
            "q4_k_m", "q4_k_s", "q5_k_m", "q5_k_s", "q6_k", "q8_0", "iq4", "iq3",
        }
        
        formatted_parts = []
        for part in parts:
            if not part:
                continue
            # Check if it's an acronym
            if part.lower() in acronyms:
                # Keep acronyms uppercase
                formatted_parts.append(part.upper() if part.lower() in {"gguf", "ggml", "gpt", "llm", "ai", "api", "cpu", "gpu", "ram", "vram"} else part.upper())
            elif part.lower().startswith("q") and any(c.isdigit() for c in part):
                # Quantization format - uppercase
                formatted_parts.append(part.upper())
            elif part.isdigit() or (part[:-1].isdigit() and part[-1] in "bB"):
                # Numbers and sizes (7b, 13B, etc) - keep as is or uppercase B
                if part[-1] in "bB" and part[:-1].isdigit():
                    formatted_parts.append(part[:-1] + "B")
                else:
                    formatted_parts.append(part)
            else:
                # Regular word - capitalize first letter
                formatted_parts.append(part.capitalize())
        
        # Join with spaces, then replace spaces with underscores for INI compatibility
        display_name = " ".join(formatted_parts)
        # Replace spaces with underscores (INI sections can't have spaces)
        return display_name.replace(" ", "_")
    
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
        
        for item in self.models_dir.iterdir():
            # Skip the models.ini file
            if item.name == "models.ini":
                continue
            
            if item.is_dir() and item.name not in valid_model_ids:
                if self._remove_path(item):
                    result.removed += 1
                    # Remove from group map
                    self._group_map.pop(item.name, None)
                else:
                    result.errors.append(f"Failed to remove orphan: {item}")
            elif item.is_file() and item.suffix == ".gguf":
                # Flat GGUF files should be removed (models go in subdirs)
                if self._remove_path(item):
                    result.removed += 1
                    logger.info("Removed flat GGUF file", path=str(item))
        
        # Update models.ini after cleanup
        if self.llama_config.generate_models_ini:
            self._update_models_ini()
        
        return result
