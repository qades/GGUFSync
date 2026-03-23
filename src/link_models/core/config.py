"""Configuration loading and management."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import yaml
from pydantic import ValidationError

from .constants import (
    DEFAULT_MODELS_SRC,
    DEFAULT_MODELS_DST,
    DEFAULT_LOCALAI_DIR,
    DEFAULT_LMSTUDIO_DIR,
    DEFAULT_OLLAMA_DIR,
    DEFAULT_TEXTGEN_DIR,
    DEFAULT_GPT4ALL_DIR,
    DEFAULT_KOBOLDCPP_DIR,
    DEFAULT_VLLM_DIR,
    DEFAULT_JAN_DIR,
    DEFAULT_LLAMA_CPP_PYTHON_DIR,
)
from .exceptions import ConfigError
from .logging import get_logger
from .models import (
    AppConfig,
    BackendConfig,
    LlamaCppConfig,
    LocalAIConfig,
    LMStudioConfig,
    OllamaConfig,
    TextGenConfig,
    GPT4AllConfig,
    KoboldCppConfig,
    vLLMConfig,
    JanConfig,
    LlamaCppPythonConfig,
    WatchConfig,
    LoggingConfig,
    SyncConfig,
)

# Alias for backward compatibility
Config = AppConfig

logger = get_logger(__name__)

# Default config file locations (in order of preference)
CONFIG_LOCATIONS = [
    Path.cwd() / "link_models.yaml",
    Path.cwd() / "link_models.yml",
    Path.home() / ".config" / "link_models" / "config.yaml",
    Path.home() / ".link_models.yaml",
    Path("/etc") / "link_models" / "config.yaml",
    Path("/etc") / "default" / "link_models",
]

# Environment variable prefix
ENV_PREFIX = "LINK_MODELS_"


def load_yaml_config(path: Path) -> dict[str, Any]:
    """Load configuration from YAML file.

    Args:
        path: Path to YAML file

    Returns:
        Configuration dictionary

    Raises:
        ConfigError: If file cannot be loaded
    """
    try:
        with open(path, "r", encoding="utf-8") as f:
            return yaml.safe_load(f) or {}
    except FileNotFoundError:
        raise ConfigError(f"Configuration file not found: {path}")
    except yaml.YAMLError as e:
        raise ConfigError(f"Invalid YAML in {path}: {e}")
    except Exception as e:
        raise ConfigError(f"Error loading {path}: {e}")


def get_env_config() -> dict[str, Any]:
    """Load configuration from environment variables.

    Environment variables are expected to be prefixed with LINK_MODELS_
    and use double underscore as nested key separator.

    Examples:
        LINK_MODELS_SOURCE_DIR=/models
        LINK_MODELS_BACKENDS__LLAMA_CPP__OUTPUT_DIR=/llama_models
        LINK_MODELS_WATCH__ENABLED=true

    Returns:
        Configuration dictionary
    """
    config: dict[str, Any] = {}

    for key, value in os.environ.items():
        if not key.startswith(ENV_PREFIX):
            continue

        # Remove prefix and split by double underscore
        config_key = key[len(ENV_PREFIX) :].lower()
        keys = config_key.split("__")

        # Parse value
        parsed_value = _parse_env_value(value)

        # Build nested structure
        current = config
        for k in keys[:-1]:
            if k not in current:
                current[k] = {}
            current = current[k]
        current[keys[-1]] = parsed_value

    return config


def _parse_env_value(value: str) -> Any:
    """Parse environment variable value to appropriate type.

    Args:
        value: String value from environment

    Returns:
        Parsed value (bool, int, float, or string)
    """
    # Boolean values
    lower = value.lower()
    if lower in ("true", "yes", "1", "on"):
        return True
    if lower in ("false", "no", "0", "off"):
        return False

    # Integer
    try:
        return int(value)
    except ValueError:
        pass

    # Float
    try:
        return float(value)
    except ValueError:
        pass

    # Path expansion
    if value.startswith("~/") or value.startswith("$HOME/"):
        return os.path.expanduser(value)
    if value.startswith("$"):
        return os.path.expandvars(value)

    # String
    return value


def merge_configs(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    """Deep merge two configuration dictionaries.

    Args:
        base: Base configuration
        override: Configuration to merge on top

    Returns:
        Merged configuration
    """
    result = dict(base)

    for key, value in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = merge_configs(result[key], value)
        else:
            result[key] = value

    return result


class ConfigLoader:
    """Loads and manages application configuration."""

    def __init__(self) -> None:
        self._config: AppConfig | None = None

    def load(
        self,
        *,
        config_path: Path | None = None,
        cli_args: dict[str, Any] | None = None,
    ) -> AppConfig:
        """Load configuration from all sources.

        Configuration is loaded in order of increasing precedence:
        1. Default values
        2. Config file
        3. Environment variables
        4. CLI arguments

        Args:
            config_path: Explicit path to config file
            cli_args: CLI argument overrides

        Returns:
            Validated AppConfig

        Raises:
            ConfigError: If configuration is invalid
        """
        # Start with empty config
        raw_config: dict[str, Any] = {}

        # 1. Load from config file if available
        file_config = self._find_and_load_config(config_path)
        if file_config:
            raw_config = merge_configs(raw_config, file_config)
            logger.debug("Loaded configuration from file", path=config_path)

        # 2. Load from environment variables
        env_config = get_env_config()
        if env_config:
            raw_config = merge_configs(raw_config, env_config)
            logger.debug("Loaded configuration from environment")

        # 3. Apply CLI arguments (highest precedence)
        if cli_args:
            # Filter out None values
            cli_config = {k: v for k, v in cli_args.items() if v is not None}
            if cli_config:
                raw_config = merge_configs(raw_config, cli_config)
                logger.debug("Applied CLI arguments")

        # 4. Parse and validate
        try:
            self._config = self._parse_config(raw_config)
        except ValidationError as e:
            raise ConfigError(
                "Configuration validation failed",
                details={"errors": e.errors()},
            ) from e

        logger.info(
            "Configuration loaded",
            source_dir=str(self._config.source_dir),
            backends=list(self._config.backends.keys()),
            watch_enabled=self._config.watch.enabled,
        )

        return self._config

    def _find_and_load_config(
        self,
        explicit_path: Path | None = None,
    ) -> dict[str, Any] | None:
        """Find and load config file.

        Args:
            explicit_path: Explicit path to config file

        Returns:
            Configuration dictionary or None
        """
        if explicit_path:
            if explicit_path.exists():
                return load_yaml_config(explicit_path)
            raise ConfigError(f"Config file not found: {explicit_path}")

        # Search in default locations
        for path in CONFIG_LOCATIONS:
            if path.exists():
                logger.debug("Found config file", path=str(path))
                return load_yaml_config(path)

        logger.debug("No config file found in default locations")
        return None

    def _parse_config(self, raw: dict[str, Any]) -> AppConfig:
        """Parse raw configuration into AppConfig.

        Args:
            raw: Raw configuration dictionary

        Returns:
            Validated AppConfig
        """
        # Handle backends specially to instantiate correct types
        backends: dict[str, BackendConfig] = {}
        raw_backends = raw.pop("backends", {})

        for name, config in raw_backends.items():
            backend_type = config.get("type", name).lower()

            if backend_type in ("llama_cpp", "llama-cpp", "llamacpp"):
                backends[name] = LlamaCppConfig(**config)
            elif backend_type == "localai":
                backends[name] = LocalAIConfig(**config)
            elif backend_type == "lmstudio":
                backends[name] = LMStudioConfig(**config)
            elif backend_type == "ollama":
                backends[name] = OllamaConfig(**config)
            elif backend_type in ("textgen", "text-generation-webui", "oobabooga"):
                backends[name] = TextGenConfig(**config)
            elif backend_type == "gpt4all":
                backends[name] = GPT4AllConfig(**config)
            elif backend_type == "koboldcpp":
                backends[name] = KoboldCppConfig(**config)
            elif backend_type == "vllm":
                backends[name] = vLLMConfig(**config)
            elif backend_type == "jan":
                backends[name] = JanConfig(**config)
            elif backend_type in ("llama_cpp_python", "llama-cpp-python", "llamacpp-python"):
                backends[name] = LlamaCppPythonConfig(**config)
            else:
                backends[name] = BackendConfig(**config)

        # Add default backends if not specified
        if not backends:
            backends["llama_cpp"] = LlamaCppConfig(
                output_dir=Path(DEFAULT_MODELS_DST),
            )
            backends["localai"] = LocalAIConfig(
                output_dir=Path(DEFAULT_LOCALAI_DIR),
            )

        # Parse nested config objects
        watch_config = WatchConfig(**raw.pop("watch", {}))
        logging_config = LoggingConfig(**raw.pop("logging", {}))
        sync_config = SyncConfig(**raw.pop("sync", {}))

        # Source directory
        source_dir = raw.pop("source_dir", Path(DEFAULT_MODELS_SRC))

        return AppConfig(
            source_dir=Path(source_dir),
            backends=backends,
            watch=watch_config,
            logging=logging_config,
            sync=sync_config,
            **raw,  # Any additional config
        )

    @property
    def config(self) -> AppConfig | None:
        """Get loaded configuration."""
        return self._config

    def generate_default_config(self) -> str:
        """Generate default configuration as YAML string.

        Returns:
            YAML configuration string
        """
        default = {
            "source_dir": DEFAULT_MODELS_SRC,
            "backends": {
                "llama_cpp": {
                    "enabled": True,
                    "output_dir": DEFAULT_MODELS_DST,
                    "generate_models_ini": True,
                    "use_subdirs": True,
                },
                "localai": {
                    "enabled": True,
                    "output_dir": DEFAULT_LOCALAI_DIR,
                    "generate_yaml": True,
                    "gpu_layers": -1,
                    "mmap": True,
                    "f16": True,
                },
                "ollama": {
                    "enabled": False,
                    "output_dir": DEFAULT_OLLAMA_DIR,
                    "generate_modelfile": True,
                },
                "textgen": {
                    "enabled": False,
                    "output_dir": DEFAULT_TEXTGEN_DIR,
                    "generate_settings_yaml": False,
                    "generate_model_configs": False,
                },
                "gpt4all": {
                    "enabled": False,
                    "output_dir": DEFAULT_GPT4ALL_DIR,
                    "generate_config": False,
                    "gpu_layers": -1,
                },
                "koboldcpp": {
                    "enabled": False,
                    "output_dir": DEFAULT_KOBOLDCPP_DIR,
                    "generate_kcpps": True,
                    "gpu_layers": -1,
                },
                "vllm": {
                    "enabled": False,
                    "output_dir": DEFAULT_VLLM_DIR,
                    "generate_config": True,
                    "trust_remote_code": True,
                },
                "jan": {
                    "enabled": False,
                    "output_dir": DEFAULT_JAN_DIR,
                    "generate_metadata": True,
                },
                "llama_cpp_python": {
                    "enabled": False,
                    "output_dir": DEFAULT_LLAMA_CPP_PYTHON_DIR,
                    "server_port": 8000,
                    "server_host": "0.0.0.0",
                    "gpu_layers": -1,
                },
            },
            "watch": {
                "enabled": False,
                "check_interval": 2.0,
                "stable_count": 3,
                "max_wait_time": 3600,
            },
            "logging": {
                "level": "INFO",
                "json_format": False,
            },
            "sync": {
                "dry_run": False,
                "prefer_hardlinks": True,
                "add_only": False,
                "preserve_orphans": False,
                "default_context_size": None,
            },
        }
        return yaml.dump(default, default_flow_style=False, sort_keys=False)
