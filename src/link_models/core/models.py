"""Data models for link_models."""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from enum import Enum, auto
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field, field_validator

from .constants import (
    DEFAULT_MODELS_DST,
    GGUF_EXTENSION,
    MULTIPART_PATTERN,
    MMPROJ_PATTERN,
)


class SyncAction(Enum):
    """Actions that can occur during synchronization."""

    CREATE = auto()
    UPDATE = auto()
    DELETE = auto()
    RESTORE = auto()
    SKIP = auto()


class SyncEventType(Enum):
    """Types of filesystem events."""

    FILE_CREATED = auto()
    FILE_MODIFIED = auto()
    FILE_DELETED = auto()
    FILE_MOVED = auto()
    DOWNLOAD_COMPLETED = auto()


@dataclass(frozen=True, slots=True)
class SyncEvent:
    """A filesystem synchronization event."""

    event_type: SyncEventType
    path: Path
    source_dir: Path
    is_partial: bool = False

    def __repr__(self) -> str:
        return (
            f"SyncEvent({self.event_type.name}, "
            f"path={self.path.name}, "
            f"source={self.source_dir.name})"
        )


@dataclass(slots=True)
class GGUFMetadata:
    """Extracted metadata from a GGUF file."""

    architecture: str | None = None
    name: str | None = None
    context_length: int | None = None
    quantization: int | None = None
    vocab_size: int | None = None
    num_hidden_layers: int | None = None
    chat_template: str | None = None
    stop_tokens: list[str] = field(default_factory=list)

    def get_backend(self) -> str:
        """Determine the appropriate backend for this model."""
        from .constants import ARCHITECTURE_BACKENDS, DEFAULT_BACKEND

        if self.architecture:
            return ARCHITECTURE_BACKENDS.get(self.architecture.lower(), DEFAULT_BACKEND)
        return DEFAULT_BACKEND

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "architecture": self.architecture,
            "name": self.name,
            "context_length": self.context_length,
            "quantization": self.quantization,
            "vocab_size": self.vocab_size,
            "num_hidden_layers": self.num_hidden_layers,
            "chat_template": self.chat_template,
            "stop_tokens": self.stop_tokens,
        }


@dataclass(slots=True)
class ModelInfo:
    """Information about a single model file."""

    path: Path
    metadata: GGUFMetadata | None = None
    file_size: int = 0
    mtime: float = 0.0

    @property
    def name(self) -> str:
        """Get the model file name."""
        return self.path.name

    @property
    def is_gguf(self) -> bool:
        """Check if this is a GGUF file."""
        return self.path.suffix.lower() == GGUF_EXTENSION

    @property
    def is_mmproj(self) -> bool:
        """Check if this is an mmproj file."""
        return MMPROJ_PATTERN in self.name.lower()

    def get_file_info(self) -> str:
        """Get file info string for comparison."""
        return f"{self.file_size}:{int(self.mtime)}"


@dataclass(slots=True)
class ModelGroup:
    """A group of related model files (multipart + mmproj)."""

    base_name: str
    files: list[ModelInfo] = field(default_factory=list)
    mmproj_file: ModelInfo | None = None
    source_dir: Path | None = None

    @property
    def is_multipart(self) -> bool:
        """Check if this is a multipart model."""
        return len(self.files) > 1

    @property
    def primary_file(self) -> ModelInfo | None:
        """Get the primary file for this group (first part or only file)."""
        if not self.files:
            return None
        # Sort to ensure first part comes first
        return sorted(self.files, key=lambda f: f.name)[0]

    @property
    def model_id(self) -> str:
        """Get normalized model ID (for filesystem/directory names)."""
        return normalize_model_id(self.base_name)

    @property
    def display_name(self) -> str:
        """Get display name preserving original capitalization.

        Uses the base_name (from filename) which preserves original capitalization.
        Metadata name is not used as it may contain unreliable or binary data.
        """
        # Always use base_name which comes from the original filename
        # This preserves the user's preferred capitalization
        return self.base_name

    @property
    def aliases(self) -> list[str]:
        """Get list of aliases for this model.

        Returns both the normalized ID and the original base name,
        allowing access by either name.
        """
        aliases = [self.model_id]  # Normalized lowercase version
        if self.base_name != self.model_id:
            aliases.append(self.base_name)  # Original capitalization
        return aliases

    @property
    def has_vision(self) -> bool:
        """Check if this model has vision capabilities (mmproj)."""
        return self.mmproj_file is not None

    def get_all_files(self) -> list[ModelInfo]:
        """Get all files including mmproj."""
        files = list(self.files)
        if self.mmproj_file:
            files.append(self.mmproj_file)
        return files


def normalize_model_id(name: str) -> str:
    """Normalize a string to a valid model ID.

    Converts to lowercase, replaces non-alphanumeric chars with hyphens,
    and removes leading/trailing hyphens.

    Args:
        name: Input string

    Returns:
        Normalized model ID
    """
    # Convert to lowercase and replace non-alphanumeric with hyphens
    result = re.sub(r"[^a-z0-9]+", "-", name.lower())
    # Remove leading/trailing hyphens and collapse multiple
    result = result.strip("-")
    result = re.sub(r"-+", "-", result)
    return result or "unknown"


def get_multipart_base(filename: str) -> str | None:
    """Extract base name from multipart filename.

    Args:
        filename: Filename to check

    Returns:
        Base name if multipart, None otherwise
    """
    match = re.search(MULTIPART_PATTERN, filename, re.IGNORECASE)
    if match:
        return filename[: match.start()]
    return None


def is_partial_download(filename: str) -> bool:
    """Check if filename indicates a partial download.

    Args:
        filename: Filename to check

    Returns:
        True if this appears to be a partial download
    """
    from .constants import PARTIAL_DOWNLOAD_EXTENSIONS

    name_lower = filename.lower()
    return any(name_lower.endswith(ext) for ext in PARTIAL_DOWNLOAD_EXTENSIONS)


def get_real_filename(filename: str) -> str:
    """Remove partial download extensions from filename.

    Args:
        filename: Filename with possible partial extension

    Returns:
        Clean filename
    """
    from .constants import PARTIAL_DOWNLOAD_EXTENSIONS

    for ext in PARTIAL_DOWNLOAD_EXTENSIONS:
        if filename.lower().endswith(ext):
            return filename[: -len(ext)]
    return filename


def strip_quantization_suffix(name: str) -> str:
    """Remove quantization suffix from model name.

    Handles both hyphen-separated and dot-separated quantization patterns:
    - model-Q4_K_M.gguf (hyphen)
    - model.v0.3.Q4_K_M.gguf (dot-separated version + quant)
    - model.Q4_K_M.gguf (dot-separated quant)

    Args:
        name: Model name possibly with quantization suffix

    Returns:
        Name without quantization suffix
    """
    # Common patterns: Q4_K_M, Q5_K_S, Q8_0, F16, BF16, IQ4_XS, MXFP4, etc.
    # Match quantization patterns with underscores like Q4_K_M, Q5_K_S, IQ4_XS
    # Use [A-Z0-9_]+ to match quant components which can include digits (e.g., Q8_0)
    patterns = [
        # Hyphen-separated quantization
        r"-(?:Q\d+_[A-Z0-9_]+|F16|BF16|IQ\d+_[A-Z0-9_]+|MXFP\d+)(?:\.gguf)?$",
        # Dot-separated quantization (e.g., v0.3.Q4_K_M or model.Q5_K_M)
        r"\.(?:Q\d+_[A-Z0-9_]+|F16|BF16|IQ\d+_[A-Z0-9_]+|MXFP\d+)(?:\.gguf)?$",
    ]
    result = name
    for pattern in patterns:
        result = re.sub(pattern, "", result, flags=re.IGNORECASE)
    return result


def get_mmproj_base(filename: str) -> str | None:
    """Extract base model name from mmproj filename.

    Detects any filename containing "mmproj" as a valid mmproj file.
    Handles various naming patterns:
    - mmproj-model-name.gguf (prefix)
    - model-name-mmproj.gguf (suffix)
    - model.name.mmproj-f32.gguf (infix with dots/dashes)
    - mmproj.gguf (standalone)

    Args:
        filename: mmproj filename

    Returns:
        Base model name or None if not an mmproj file
    """
    name_lower = filename.lower()

    # Quick check: filename must contain "mmproj" to be considered
    if "mmproj" not in name_lower:
        return None

    # Pattern: mmproj-model-name.gguf or mmproj-model-name-Q4_K_M.gguf
    if name_lower.startswith("mmproj-"):
        base = filename[7:]  # Remove "mmproj-" prefix
        # Remove quantization suffix
        base = strip_quantization_suffix(base)
        # Remove precision suffix (e.g., -f32, -f16, -fp32, -bf16)
        # Include optional .gguf extension since it hasn't been removed yet
        base = re.sub(r"-(?:f32|f16|fp32|bf16|fp16)(?:\.gguf)?$", "", base, flags=re.IGNORECASE)
        # Remove .gguf extension
        base = base.replace(GGUF_EXTENSION, "")
        return base

    # Pattern: model-name-mmproj.gguf or model-name-mmproj-Q4_K_M.gguf
    if "-mmproj" in name_lower:
        # Find the position of -mmproj in the original filename (preserve case)
        mmproj_pos = name_lower.find("-mmproj")
        base = filename[:mmproj_pos]
        # Remove trailing dashes/underscores
        base = base.rstrip("-_")
        # Also remove any trailing quantization suffix
        base = strip_quantization_suffix(base)
        return base

    # Pattern: model.name.mmproj-f32.gguf (mmproj with any separator)
    # Extract everything before "mmproj" and clean it up
    mmproj_pos = name_lower.find("mmproj")
    if mmproj_pos > 0:
        # Get everything before "mmproj" from the ORIGINAL filename (preserve case)
        base = filename[:mmproj_pos]
        # Remove trailing dots, dashes, underscores
        base = base.rstrip(".-_")
        # Remove quantization suffix if present
        base = strip_quantization_suffix(base)
        return base if base else None

    # Pattern: mmproj.gguf in a model directory (standalone mmproj)
    if name_lower == "mmproj.gguf":
        # Return empty string to indicate it's a generic mmproj
        # The caller should use the directory name to match
        return ""

    return None


# Pydantic models for configuration


class BackendConfig(BaseModel):
    """Configuration for a backend."""

    enabled: bool = True
    output_dir: Path
    extra_params: dict[str, Any] = Field(default_factory=dict)
    prefer_hardlinks: bool = True
    ignore_file: Path | None = Field(default=None)

    @field_validator("output_dir")
    @classmethod
    def validate_output_dir(cls, v: Path) -> Path:
        return v.expanduser().resolve()

    @field_validator("ignore_file")
    @classmethod
    def validate_ignore_file(cls, v: Path | None) -> Path | None:
        if v is not None:
            return v.expanduser().resolve()
        return v


class LlamaCppConfig(BackendConfig):
    """Configuration for llama.cpp backend."""

    generate_models_ini: bool = True
    models_ini_path: Path | None = None
    use_subdirs: bool = True


class LocalAIConfig(BackendConfig):
    """Configuration for LocalAI backend."""

    generate_yaml: bool = True
    yaml_prefix: str = "model-"
    gpu_layers: int = -1
    mmap: bool = True
    f16: bool = True


class LMStudioConfig(BackendConfig):
    """Configuration for LM Studio backend."""

    generate_manifest: bool = True


class OllamaConfig(BackendConfig):
    """Configuration for Ollama backend.

    Ollama stores models in a specific format with manifests.
    This backend creates symlinks and optional Modelfiles.
    """

    generate_modelfile: bool = True
    manifest_template: str = "default"
    additional_params: dict[str, Any] = Field(default_factory=dict)


class TextGenConfig(BackendConfig):
    """Configuration for Text Generation WebUI (oobabooga) backend.

    TextGen uses a flat directory structure in user_data/models.
    """

    generate_settings_yaml: bool = False
    generate_model_configs: bool = False
    model_config_template: Path | None = None
    settings_template: Path | None = None


class GPT4AllConfig(BackendConfig):
    """Configuration for GPT4All backend.

    GPT4All uses a flat directory structure with model files.
    It can optionally use a config file for each model.
    """

    generate_config: bool = False
    config_template: Path | None = None
    default_context_size: int = 4096
    default_gpu_layers: int = -1


class KoboldCppConfig(BackendConfig):
    """Configuration for KoboldCpp backend.

    KoboldCpp uses .kcpps sidecar config files alongside model files.
    """

    generate_kcpps: bool = True
    default_context_size: int = 4096
    default_gpu_layers: int = -1
    default_threads: int = 5


class vLLMConfig(BackendConfig):
    """Configuration for vLLM backend.

    vLLM uses HuggingFace-style directory structure.
    """

    generate_config: bool = True
    config_template: Path | None = None
    trust_remote_code: bool = True
    enforce_eager: bool = False


class IgnoreConfig(BaseModel):
    """Configuration for per-backend model filtering."""

    enabled: bool = True
    ignore_file: Path = Field(default_factory=lambda: Path(".linkmodelsignore"))

    @field_validator("ignore_file")
    @classmethod
    def validate_ignore_file(cls, v: Path) -> Path:
        return v.expanduser().resolve()


class WatchConfig(BaseModel):
    """Configuration for filesystem watching."""

    enabled: bool = False
    check_interval: float = 2.0
    stable_count: int = 3
    max_wait_time: int = 3600
    recursive: bool = True

    @field_validator("check_interval")
    @classmethod
    def validate_check_interval(cls, v: float) -> float:
        if v <= 0:
            raise ValueError("check_interval must be positive")
        if v < 0.1:
            raise ValueError("check_interval must be at least 0.1 seconds")
        return v

    @field_validator("stable_count")
    @classmethod
    def validate_stable_count(cls, v: int) -> int:
        if v < 1:
            raise ValueError("stable_count must be at least 1")
        if v > 100:
            raise ValueError("stable_count must be at most 100")
        return v

    @field_validator("max_wait_time")
    @classmethod
    def validate_max_wait_time(cls, v: int) -> int:
        if v < 1:
            raise ValueError("max_wait_time must be at least 1 second")
        if v > 86400:
            raise ValueError("max_wait_time must be at most 86400 seconds (1 day)")
        return v


class LoggingConfig(BaseModel):
    """Configuration for logging."""

    level: str = "INFO"
    json_format: bool = False
    file: Path | None = None

    @field_validator("level")
    @classmethod
    def validate_level(cls, v: str) -> str:
        allowed = {"DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"}
        v_upper = v.upper()
        if v_upper not in allowed:
            raise ValueError(f"Invalid log level: {v}")
        return v_upper


class SyncConfig(BaseModel):
    """Configuration for synchronization behavior."""

    dry_run: bool = False
    preserve_orphans: bool = False  # If True, don't remove files not in source
    follow_symlinks: bool = False
    prefer_hardlinks: bool = True
    add_only: bool = False  # If True, never delete from backends, only add
    global_ignore_file: Path | None = Field(default=None)

    @field_validator("global_ignore_file")
    @classmethod
    def validate_ignore_file(cls, v: Path | None) -> Path | None:
        if v is not None:
            return v.expanduser().resolve()
        return v


class AppConfig(BaseModel):
    """Complete application configuration."""

    source_dir: Path = Field(default=Path("/models"))
    backends: dict[str, BackendConfig] = Field(default_factory=dict)
    watch: WatchConfig = Field(default_factory=WatchConfig)
    logging: LoggingConfig = Field(default_factory=LoggingConfig)
    sync: SyncConfig = Field(default_factory=SyncConfig)

    @field_validator("source_dir")
    @classmethod
    def validate_source_dir(cls, v: Path) -> Path:
        return v.expanduser().resolve()

    model_config = {
        "extra": "allow",
    }
