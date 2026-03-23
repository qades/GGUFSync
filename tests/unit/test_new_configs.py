"""Unit tests for new backend configurations."""

from __future__ import annotations

from pathlib import Path

import pytest

from link_models.core.models import (
    OllamaConfig,
    TextGenConfig,
    GPT4AllConfig,
    KoboldCppConfig,
    vLLMConfig,
    SyncConfig,
    BackendConfig,
    AppConfig,
    WatchConfig,
    LoggingConfig,
)


class TestOllamaConfig:
    """Tests for OllamaConfig."""

    def test_default_values(self, temp_dir: Path) -> None:
        """Test default configuration values."""
        config = OllamaConfig(output_dir=temp_dir / "models")

        assert config.enabled is True
        assert config.generate_modelfile is True
        assert config.additional_params == {}

    def test_custom_values(self, temp_dir: Path) -> None:
        """Test custom configuration values."""
        config = OllamaConfig(
            output_dir=temp_dir / "models",
            enabled=False,
            generate_modelfile=False,
            additional_params={"temperature": 0.7},
        )

        assert config.enabled is False
        assert config.generate_modelfile is False
        assert config.additional_params["temperature"] == 0.7

    def test_inherits_from_backend_config(self, temp_dir: Path) -> None:
        """Test that OllamaConfig inherits from BackendConfig."""
        config = OllamaConfig(output_dir=temp_dir / "models")

        assert hasattr(config, "prefer_hardlinks")
        assert hasattr(config, "ignore_file")


class TestTextGenConfig:
    """Tests for TextGenConfig."""

    def test_default_values(self, temp_dir: Path) -> None:
        """Test default configuration values."""
        config = TextGenConfig(output_dir=temp_dir / "models")

        assert config.enabled is True
        assert config.generate_settings_yaml is False
        assert config.generate_model_configs is False

    def test_custom_values(self, temp_dir: Path) -> None:
        """Test custom configuration values."""
        config = TextGenConfig(
            output_dir=temp_dir / "models",
            generate_settings_yaml=True,
            generate_model_configs=True,
        )

        assert config.generate_settings_yaml is True
        assert config.generate_model_configs is True


class TestGPT4AllConfig:
    """Tests for GPT4AllConfig."""

    def test_default_values(self, temp_dir: Path) -> None:
        """Test default configuration values."""
        config = GPT4AllConfig(output_dir=temp_dir / "models")

        assert config.generate_config is False
        assert config.default_context_size == 4096
        assert config.default_gpu_layers == -1

    def test_custom_values(self, temp_dir: Path) -> None:
        """Test custom configuration values."""
        config = GPT4AllConfig(
            output_dir=temp_dir / "models",
            generate_config=True,
            default_context_size=8192,
            default_gpu_layers=32,
        )

        assert config.generate_config is True
        assert config.default_context_size == 8192
        assert config.default_gpu_layers == 32


class TestKoboldCppConfig:
    """Tests for KoboldCppConfig."""

    def test_default_values(self, temp_dir: Path) -> None:
        """Test default configuration values."""
        config = KoboldCppConfig(output_dir=temp_dir / "models")

        assert config.generate_kcpps is True
        assert config.default_context_size == 4096
        assert config.default_gpu_layers == -1
        assert config.default_threads == 5

    def test_custom_values(self, temp_dir: Path) -> None:
        """Test custom configuration values."""
        config = KoboldCppConfig(
            output_dir=temp_dir / "models",
            generate_kcpps=True,
            default_context_size=16384,
            default_gpu_layers=64,
            default_threads=8,
        )

        assert config.default_context_size == 16384
        assert config.default_gpu_layers == 64
        assert config.default_threads == 8


class TestvLLMConfig:
    """Tests for vLLMConfig."""

    def test_default_values(self, temp_dir: Path) -> None:
        """Test default configuration values."""
        config = vLLMConfig(output_dir=temp_dir / "models")

        assert config.generate_config is True
        assert config.trust_remote_code is True
        assert config.enforce_eager is False

    def test_custom_values(self, temp_dir: Path) -> None:
        """Test custom configuration values."""
        config = vLLMConfig(
            output_dir=temp_dir / "models",
            trust_remote_code=False,
            enforce_eager=True,
        )

        assert config.trust_remote_code is False
        assert config.enforce_eager is True


class TestSyncConfig:
    """Tests for SyncConfig with new options."""

    def test_add_only_default(self) -> None:
        """Test add_only defaults to False."""
        config = SyncConfig()

        assert config.add_only is False

    def test_add_only_enabled(self) -> None:
        """Test add_only can be enabled."""
        config = SyncConfig(add_only=True)

        assert config.add_only is True

    def test_global_ignore_file_default(self) -> None:
        """Test global_ignore_file defaults to None."""
        config = SyncConfig()

        assert config.global_ignore_file is None

    def test_global_ignore_file_set(self, temp_dir: Path) -> None:
        """Test global_ignore_file can be set."""
        ignore_path = temp_dir / "ignore.txt"
        config = SyncConfig(global_ignore_file=ignore_path)

        assert config.global_ignore_file == ignore_path

    def test_global_ignore_file_expands_path(self, temp_dir: Path) -> None:
        """Test that global_ignore_file expands user paths."""
        config = SyncConfig(global_ignore_file=Path("~/ignore.txt"))

        # Should be expanded to absolute path
        assert config.global_ignore_file.is_absolute()


class TestBackendConfig:
    """Tests for BackendConfig with ignore_file."""

    def test_ignore_file_default(self, temp_dir: Path) -> None:
        """Test ignore_file defaults to None."""
        config = BackendConfig(output_dir=temp_dir / "models")

        assert config.ignore_file is None

    def test_ignore_file_set(self, temp_dir: Path) -> None:
        """Test ignore_file can be set."""
        ignore_path = temp_dir / "ignore.txt"
        config = BackendConfig(
            output_dir=temp_dir / "models",
            ignore_file=ignore_path,
        )

        assert config.ignore_file == ignore_path


class TestAppConfigWithNewBackends:
    """Tests for AppConfig with new backend types."""

    def test_ollama_backend_in_config(self, temp_dir: Path) -> None:
        """Test that Ollama backend can be configured."""
        config = AppConfig(
            source_dir=temp_dir / "models",
            backends={
                "ollama": OllamaConfig(output_dir=temp_dir / "ollama"),
            },
            watch=WatchConfig(),
            logging=LoggingConfig(),
            sync=SyncConfig(),
        )

        assert "ollama" in config.backends
        assert isinstance(config.backends["ollama"], OllamaConfig)

    def test_multiple_new_backends(self, temp_dir: Path) -> None:
        """Test multiple new backends in config."""
        config = AppConfig(
            source_dir=temp_dir / "models",
            backends={
                "ollama": OllamaConfig(output_dir=temp_dir / "ollama"),
                "koboldcpp": KoboldCppConfig(output_dir=temp_dir / "kobold"),
                "vllm": vLLMConfig(output_dir=temp_dir / "vllm"),
            },
            watch=WatchConfig(),
            logging=LoggingConfig(),
            sync=SyncConfig(),
        )

        assert len(config.backends) == 3
        assert isinstance(config.backends["ollama"], OllamaConfig)
        assert isinstance(config.backends["koboldcpp"], KoboldCppConfig)
        assert isinstance(config.backends["vllm"], vLLMConfig)
