"""Integration tests for new backends."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import yaml

from link_models.backends import (
    OllamaBackend,
    TextGenBackend,
    GPT4AllBackend,
    KoboldCppBackend,
    vLLMBackend,
)
from link_models.core.models import (
    OllamaConfig,
    TextGenConfig,
    GPT4AllConfig,
    KoboldCppConfig,
    vLLMConfig,
    ModelGroup,
    ModelInfo,
)


class TestOllamaBackend:
    """Integration tests for OllamaBackend."""

    @pytest.fixture
    def backend(self, temp_dir: Path) -> OllamaBackend:
        config = OllamaConfig(
            output_dir=temp_dir / "ollama_models",
            generate_modelfile=True,
        )
        return OllamaBackend(config)

    @pytest.fixture
    def model_group(self, temp_dir: Path) -> ModelGroup:
        source_file = temp_dir / "source" / "test-model.gguf"
        source_file.parent.mkdir()
        source_file.write_bytes(b"GGUF content")

        return ModelGroup(
            base_name="test-model",
            files=[ModelInfo(path=source_file)],
            source_dir=source_file.parent,
        )

    def test_setup_creates_directories(self, backend: OllamaBackend) -> None:
        backend.setup()
        assert backend.output_dir.exists()
        assert backend.manifests_dir.exists()

    def test_sync_creates_model_directory(
        self,
        backend: OllamaBackend,
        model_group: ModelGroup,
    ) -> None:
        backend.setup()
        result = backend.sync_group(model_group, model_group.source_dir)

        assert result.success is True
        assert result.linked >= 1

        expected_dir = backend.models_dir / "test-model"
        assert expected_dir.exists()

    def test_sync_generates_modelfile(
        self,
        backend: OllamaBackend,
        model_group: ModelGroup,
    ) -> None:
        backend.setup()
        backend.sync_group(model_group, model_group.source_dir)

        modelfile_path = backend.models_dir / "test-model" / "Modelfile"
        assert modelfile_path.exists()

        content = modelfile_path.read_text()
        assert "FROM" in content
        assert "test-model.gguf" in content

    def test_remove_group_removes_directory(
        self,
        backend: OllamaBackend,
        model_group: ModelGroup,
    ) -> None:
        backend.setup()
        backend.sync_group(model_group, model_group.source_dir)

        result = backend.remove_group("test-model")

        assert result.removed >= 1
        assert not (backend.models_dir / "test-model").exists()


class TestTextGenBackend:
    """Integration tests for TextGenBackend."""

    @pytest.fixture
    def backend(self, temp_dir: Path) -> TextGenBackend:
        config = TextGenConfig(
            output_dir=temp_dir / "textgen_models",
            generate_model_configs=True,
        )
        return TextGenBackend(config)

    @pytest.fixture
    def model_group(self, temp_dir: Path) -> ModelGroup:
        source_file = temp_dir / "source" / "llama3-8b.gguf"
        source_file.parent.mkdir()
        source_file.write_bytes(b"GGUF content")

        return ModelGroup(
            base_name="llama3-8b",
            files=[ModelInfo(path=source_file)],
            source_dir=source_file.parent,
        )

    def test_setup_creates_directory(self, backend: TextGenBackend) -> None:
        backend.setup()
        assert backend.output_dir.exists()
        assert backend.configs_dir.exists()

    def test_sync_creates_model_subdirectory(
        self,
        backend: TextGenBackend,
        model_group: ModelGroup,
    ) -> None:
        backend.setup()
        result = backend.sync_group(model_group, model_group.source_dir)

        assert result.success is True

        expected_dir = backend.models_dir / "llama3-8b"
        assert expected_dir.exists()

    def test_sync_generates_model_config(
        self,
        backend: TextGenBackend,
        model_group: ModelGroup,
    ) -> None:
        backend.setup()
        backend.sync_group(model_group, model_group.source_dir)

        config_path = backend.configs_dir / "llama3-8b.yaml"
        assert config_path.exists()

        with open(config_path) as f:
            config = yaml.safe_load(f)

        assert config["model_name"] == "llama3-8b"
        assert config["llm_loader"] == "llama.cpp"


class TestGPT4AllBackend:
    """Integration tests for GPT4AllBackend."""

    @pytest.fixture
    def backend(self, temp_dir: Path) -> GPT4AllBackend:
        config = GPT4AllConfig(
            output_dir=temp_dir / "gpt4all_models",
            generate_config=True,
        )
        return GPT4AllBackend(config)

    @pytest.fixture
    def model_group(self, temp_dir: Path) -> ModelGroup:
        source_file = temp_dir / "source" / "mistral-7b.gguf"
        source_file.parent.mkdir()
        source_file.write_bytes(b"GGUF content")

        return ModelGroup(
            base_name="mistral-7b",
            files=[ModelInfo(path=source_file)],
            source_dir=source_file.parent,
        )

    def test_setup_creates_directory(self, backend: GPT4AllBackend) -> None:
        backend.setup()
        assert backend.output_dir.exists()

    def test_sync_creates_model_directory(
        self,
        backend: GPT4AllBackend,
        model_group: ModelGroup,
    ) -> None:
        backend.setup()
        result = backend.sync_group(model_group, model_group.source_dir)

        assert result.success is True

        expected_dir = backend.models_dir / "mistral-7b"
        assert expected_dir.exists()

    def test_sync_generates_json_config(
        self,
        backend: GPT4AllBackend,
        model_group: ModelGroup,
    ) -> None:
        backend.setup()
        backend.sync_group(model_group, model_group.source_dir)

        config_path = backend.configs_dir / "mistral-7b.json"
        assert config_path.exists()

        with open(config_path) as f:
            config = json.load(f)

        assert "model" in config
        assert "parameters" in config


class TestKoboldCppBackend:
    """Integration tests for KoboldCppBackend."""

    @pytest.fixture
    def backend(self, temp_dir: Path) -> KoboldCppBackend:
        config = KoboldCppConfig(
            output_dir=temp_dir / "koboldcpp_models",
            generate_kcpps=True,
            default_context_size=4096,
            default_gpu_layers=32,
        )
        return KoboldCppBackend(config)

    @pytest.fixture
    def model_group(self, temp_dir: Path) -> ModelGroup:
        source_file = temp_dir / "source" / "qwen-14b.gguf"
        source_file.parent.mkdir()
        source_file.write_bytes(b"GGUF content")

        return ModelGroup(
            base_name="qwen-14b",
            files=[ModelInfo(path=source_file)],
            source_dir=source_file.parent,
        )

    def test_setup_creates_directory(self, backend: KoboldCppBackend) -> None:
        backend.setup()
        assert backend.output_dir.exists()

    def test_sync_creates_model_directory(
        self,
        backend: KoboldCppBackend,
        model_group: ModelGroup,
    ) -> None:
        backend.setup()
        result = backend.sync_group(model_group, model_group.source_dir)

        assert result.success is True

        expected_dir = backend.models_dir / "qwen-14b"
        assert expected_dir.exists()

    def test_sync_generates_kcpps_file(
        self,
        backend: KoboldCppBackend,
        model_group: ModelGroup,
    ) -> None:
        backend.setup()
        backend.sync_group(model_group, model_group.source_dir)

        kcpps_path = backend.models_dir / "qwen-14b.kcpps"
        assert kcpps_path.exists()

        with open(kcpps_path) as f:
            config = json.load(f)

        assert "model_param" in config
        assert "contextsize" in config
        assert config["contextsize"] == 4096
        assert config["gpulayers"] == 32


class TestvLLMBackend:
    """Integration tests for vLLMBackend."""

    @pytest.fixture
    def backend(self, temp_dir: Path) -> vLLMBackend:
        config = vLLMConfig(
            output_dir=temp_dir / "vllm_models",
            generate_config=True,
            trust_remote_code=True,
        )
        return vLLMBackend(config)

    @pytest.fixture
    def model_group(self, temp_dir: Path) -> ModelGroup:
        source_file = temp_dir / "source" / "phi-3.gguf"
        source_file.parent.mkdir()
        source_file.write_bytes(b"GGUF content")

        return ModelGroup(
            base_name="phi-3",
            files=[ModelInfo(path=source_file)],
            source_dir=source_file.parent,
        )

    def test_setup_creates_directory(self, backend: vLLMBackend) -> None:
        backend.setup()
        assert backend.output_dir.exists()

    def test_sync_creates_model_directory(
        self,
        backend: vLLMBackend,
        model_group: ModelGroup,
    ) -> None:
        backend.setup()
        result = backend.sync_group(model_group, model_group.source_dir)

        assert result.success is True

        expected_dir = backend.models_dir / "phi-3"
        assert expected_dir.exists()

    def test_sync_generates_config_json(
        self,
        backend: vLLMBackend,
        model_group: ModelGroup,
    ) -> None:
        backend.setup()
        backend.sync_group(model_group, model_group.source_dir)

        config_path = backend.models_dir / "phi-3" / "config.json"
        assert config_path.exists()

        with open(config_path) as f:
            config = json.load(f)

        assert "model_type" in config
        assert config["trust_remote_code"] is True


class TestBackendCleanup:
    """Tests for orphan cleanup in new backends."""

    def test_ollama_cleanup_removes_orphans(self, temp_dir: Path) -> None:
        """Test that Ollama backend cleans up orphaned directories."""
        backend = OllamaBackend(OllamaConfig(output_dir=temp_dir / "ollama"))
        backend.setup()

        # Create an orphaned directory
        orphan_dir = backend.models_dir / "orphan-model"
        orphan_dir.mkdir()
        (orphan_dir / "model.gguf").write_bytes(b"GGUF")

        # Run cleanup with valid set that doesn't include orphan
        result = backend.cleanup_orphans({"other-model"})

        assert result.removed >= 1
        assert not orphan_dir.exists()

    def test_koboldcpp_cleanup_removes_orphans_and_kcpps(
        self,
        temp_dir: Path,
    ) -> None:
        """Test that KoboldCpp cleans up orphaned directories and .kcpps files."""
        backend = KoboldCppBackend(KoboldCppConfig(output_dir=temp_dir / "kobold"))
        backend.setup()

        # Create orphaned model dir and .kcpps file
        orphan_dir = backend.models_dir / "orphan-model"
        orphan_dir.mkdir()
        (orphan_dir / "model.gguf").write_bytes(b"GGUF")

        orphan_kcpps = backend.models_dir / "orphan-model.kcpps"
        orphan_kcpps.write_text('{"model_param": "test"}')

        # Run cleanup
        result = backend.cleanup_orphans({"other-model"})

        assert result.removed >= 2  # Both dir and kcpps
        assert not orphan_dir.exists()
        assert not orphan_kcpps.exists()
