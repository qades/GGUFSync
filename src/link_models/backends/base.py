"""Base class for all backends."""

from __future__ import annotations

import os
import shutil
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from ..core.constants import DIR_PERMISSIONS, FILE_PERMISSIONS
from ..core.logging import get_logger, log_action
from ..core.models import BackendConfig, ModelGroup, SyncAction

logger = get_logger(__name__)


@dataclass
class LinkResult:
    """Result of a single link operation."""
    success: bool
    action: SyncAction
    source: Path
    target: Path
    is_hardlink: bool = False
    error: str | None = None


@dataclass
class BackendResult:
    """Result of a backend sync operation."""
    success: bool
    linked: int = 0
    updated: int = 0
    removed: int = 0
    skipped: int = 0
    errors: list[str] = field(default_factory=list)
    details: dict[str, Any] = field(default_factory=dict)
    skip_reasons: list[dict[str, str]] = field(default_factory=list)


class Backend(ABC):
    """Abstract base class for all backends."""
    
    def __init__(self, config: BackendConfig) -> None:
        """Initialize backend.
        
        Args:
            config: Backend configuration
        """
        self.config = config
        self.output_dir = config.output_dir
        self.logger = get_logger(self.__class__.__name__)
    
    @property
    @abstractmethod
    def name(self) -> str:
        """Return the backend name."""
        pass
    
    @abstractmethod
    def sync_group(self, group: ModelGroup, source_dir: Path) -> BackendResult:
        """Sync a model group to this backend.
        
        Args:
            group: Model group to sync
            source_dir: Source directory (ground truth)
            
        Returns:
            BackendResult with operation results
        """
        pass
    
    @abstractmethod
    def remove_group(self, model_id: str) -> BackendResult:
        """Remove a model group from this backend.
        
        Args:
            model_id: Normalized model ID to remove
            
        Returns:
            BackendResult with operation results
        """
        pass
    
    def setup(self) -> None:
        """Setup the backend (create directories, etc.)."""
        if not self.config.enabled:
            return
        
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self._set_permissions(self.output_dir)
        self.logger.debug("Backend setup complete", output_dir=str(self.output_dir))
    
    def cleanup(self) -> None:
        """Cleanup any resources. Called during shutdown."""
        pass
    
    def _ensure_dir(self, path: Path) -> None:
        """Ensure a directory exists with proper permissions.
        
        Args:
            path: Directory path to ensure exists
        """
        if not path.exists():
            path.mkdir(parents=True, exist_ok=True)
            self._set_permissions(path)
    
    def _set_permissions(self, path: Path) -> None:
        """Set appropriate permissions on a file or directory.
        
        Args:
            path: Path to set permissions on
        """
        try:
            if path.is_dir():
                path.chmod(DIR_PERMISSIONS)
            else:
                path.chmod(FILE_PERMISSIONS)
        except OSError as e:
            self.logger.warning("Failed to set permissions", path=str(path), error=str(e))
    
    def _create_link(
        self,
        source: Path,
        target: Path,
        *,
        dry_run: bool = False,
        prefer_hardlink: bool = True,
    ) -> LinkResult:
        """Create a link from source to target.
        
        Args:
            source: Source file (must exist)
            target: Target path to create
            dry_run: If True, don't actually create the link
            prefer_hardlink: Try hardlink first, fallback to symlink
            
        Returns:
            LinkResult with operation details
        """
        if not source.exists():
            return LinkResult(
                success=False,
                action=SyncAction.SKIP,
                source=source,
                target=target,
                error=f"Source does not exist: {source}",
            )
        
        # Ensure target directory exists
        if not dry_run:
            self._ensure_dir(target.parent)
        
        # Check if target already exists and is up to date
        target_existed = target.exists() or target.is_symlink()
        if target_existed:
            if self._is_same_file(source, target):
                return LinkResult(
                    success=True,
                    action=SyncAction.SKIP,
                    source=source,
                    target=target,
                    is_hardlink=self._is_hardlink(target),
                )
            
            if not dry_run:
                # Remove existing file/link
                if target.is_dir():
                    shutil.rmtree(target)
                else:
                    target.unlink()
        
        if dry_run:
            log_action(
                self.logger,
                "link",
                f"{source.name} -> {target}",
                dry_run=True,
                source=str(source),
                target=str(target),
            )
            return LinkResult(
                success=True,
                action=SyncAction.UPDATE if target_existed else SyncAction.CREATE,
                source=source,
                target=target,
                is_hardlink=prefer_hardlink,
            )
        
        # Try to create link
        is_hardlink = False
        try:
            if prefer_hardlink:
                try:
                    os.link(source, target)  # Hardlink
                    is_hardlink = True
                    action = "HARDLINK"
                except OSError:
                    # Cross-device or other error, try symlink
                    target.symlink_to(source)
                    action = "SYMLINK"
            else:
                target.symlink_to(source)
                action = "SYMLINK"
            
            self._set_permissions(target)
            
            log_action(
                self.logger,
                action,
                f"{source.name} -> {target}",
                source=str(source),
                target=str(target),
            )
            
            return LinkResult(
                success=True,
                action=SyncAction.UPDATE if target_existed else SyncAction.CREATE,
                source=source,
                target=target,
                is_hardlink=is_hardlink,
            )
        
        except OSError as e:
            error_msg = f"Failed to create link: {e}"
            self.logger.error(error_msg, source=str(source), target=str(target))
            return LinkResult(
                success=False,
                action=SyncAction.SKIP,
                source=source,
                target=target,
                error=error_msg,
            )
    
    def _is_same_file(self, path1: Path, path2: Path) -> bool:
        """Check if two paths point to the same file.
        
        Compares by inode if on same filesystem, otherwise by size and mtime.
        
        Args:
            path1: First path
            path2: Second path
            
        Returns:
            True if files are the same
        """
        try:
            # Try inode comparison first (fastest)
            stat1 = path1.stat()
            stat2 = path2.stat()
            
            if stat1.st_ino == stat2.st_ino and stat1.st_dev == stat2.st_dev:
                return True
            
            # Fallback to size and mtime comparison
            if stat1.st_size != stat2.st_size:
                return False
            
            # If size matches and mtime matches, assume same file
            return int(stat1.st_mtime) == int(stat2.st_mtime)
        
        except (OSError, FileNotFoundError):
            return False
    
    def _is_hardlink(self, path: Path) -> bool:
        """Check if path is a hardlink (not a symlink).
        
        Args:
            path: Path to check
            
        Returns:
            True if path is a hardlink (has multiple links)
        """
        try:
            return path.exists() and not path.is_symlink() and path.stat().st_nlink > 1
        except OSError:
            return False
    
    def _remove_path(self, path: Path, *, dry_run: bool = False) -> bool:
        """Remove a file or directory.
        
        Args:
            path: Path to remove
            dry_run: If True, don't actually remove
            
        Returns:
            True if removal succeeded or path didn't exist
        """
        if not path.exists() and not path.is_symlink():
            return True
        
        if dry_run:
            log_action(
                self.logger,
                "remove",
                str(path),
                dry_run=True,
                path=str(path),
            )
            return True
        
        try:
            if path.is_dir():
                shutil.rmtree(path)
            else:
                path.unlink()
            
            self.logger.info("REMOVED", path=str(path))
            return True
        
        except OSError as e:
            self.logger.error("Failed to remove", path=str(path), error=str(e))
            return False
