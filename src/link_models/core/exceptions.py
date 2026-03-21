"""Custom exceptions for link_models."""

from __future__ import annotations


class LinkModelsError(Exception):
    """Base exception for all link_models errors."""
    
    def __init__(self, message: str, *, details: dict[str, object] | None = None) -> None:
        super().__init__(message)
        self.message = message
        self.details = details or {}


class ConfigError(LinkModelsError):
    """Configuration-related errors."""
    pass


class GGUFError(LinkModelsError):
    """GGUF parsing errors."""
    pass


class SyncError(LinkModelsError):
    """File synchronization errors."""
    pass


class BackendError(LinkModelsError):
    """Backend-specific errors."""
    
    def __init__(
        self,
        message: str,
        *,
        backend_name: str | None = None,
        details: dict[str, object] | None = None,
    ) -> None:
        super().__init__(message, details=details)
        self.backend_name = backend_name


class WatchError(LinkModelsError):
    """Filesystem watching errors."""
    pass


class ServiceError(LinkModelsError):
    """Service installation/management errors."""
    pass
