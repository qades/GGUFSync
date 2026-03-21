"""Integration tests for backends."""

from __future__ import annotations

from pathlib import Path

import pytest

from link_models.backends import LlamaCppBackend, LocalAIBackend, LMStudioBackend
from link_models.core.models import (
    LlamaCppConfig,
    LocalAIConfig,
    LMStudioConfig,
    ModelGroup,
    ModelInfo,
    SyncAction,
)


class TestLlamaCppBackend:
    """Integration tests for LlamaCppBackend."""
    
    @pytest.fixture
    def backend(self, temp_dir: Path) -> LlamaCppBackend:
        config = LlamaCppConfig(output_dir=temp_dir / "llama_models")
        return LlamaCppBackend(config)
    
    @pytest.fixture
    def model_group(self, temp_dir: Path) -> ModelGroup:
        source_file = temp_dir / "source" / "model.gguf"
        source_file.parent.mkdir()
        source_file.write_bytes(b"GGUF content")
        
        return ModelGroup(
            base_name="test-model",
            files=[ModelInfo(path=source_file)],
            source_dir=source_file.parent,
        )
    
    def test_setup_creates_directory(self, backend: LlamaCppBackend) -> None:
        backend.setup()
        assert backend.output_dir.exists()
    
    def test_sync_group_creates_hardlink(
        self,
        backend: LlamaCppBackend,
        model_group: ModelGroup,
    ) -> None:
        backend.setup()
        
        result = backend.sync_group(model_group, model_group.source_dir)
        
        assert result.success is True
        assert result.linked == 1
        
        # Check file was created in subdirectory
        expected_file = backend.models_dir / "test-model" / "model.gguf"
        assert expected_file.exists()
    
    def test_sync_group_skips_existing_up_to_date(
        self,
        backend: LlamaCppBackend,
        model_group: ModelGroup,
    ) -> None:
        backend.setup()
        
        # First sync
        backend.sync_group(model_group, model_group.source_dir)
        
        # Second sync should skip
        result = backend.sync_group(model_group, model_group.source_dir)
        
        assert result.skipped == 1
    
    def test_remove_group_deletes_directory(
        self,
        backend: LlamaCppBackend,
        model_group: ModelGroup,
    ) -> None:
        backend.setup()
        backend.sync_group(model_group, model_group.source_dir)
        
        result = backend.remove_group("test-model")
        
        assert result.removed == 1
        assert not (backend.models_dir / "test-model").exists()
    
    def test_cleanup_orphans_removes_extra_directories(
        self,
        backend: LlamaCppBackend,
        temp_dir: Path,
    ) -> None:
        backend.setup()
        
        # Create extra directory
        extra_dir = backend.models_dir / "orphan-model"
        extra_dir.mkdir()
        (extra_dir / "file.gguf").write_text("content")
        
        result = backend.cleanup_orphans({"valid-model"})
        
        assert result.removed == 1
        assert not extra_dir.exists()


class TestLocalAIBackend:
    """Integration tests for LocalAIBackend."""
    
    @pytest.fixture
    def backend(self, temp_dir: Path) -> LocalAIBackend:
        config = LocalAIConfig(output_dir=temp_dir / "localai_models")
        return LocalAIBackend(config)
    
    @pytest.fixture
    def model_group(self, temp_dir: Path) -> ModelGroup:
        source_file = temp_dir / "source" / "model.gguf"
        source_file.parent.mkdir()
        source_file.write_bytes(b"GGUF content")
        
        return ModelGroup(
            base_name="test-model",
            files=[ModelInfo(path=source_file)],
            source_dir=source_file.parent,
        )
    
    def test_setup_creates_directory(self, backend: LocalAIBackend) -> None:
        backend.setup()
        assert backend.output_dir.exists()
    
    def test_sync_group_creates_yaml_config(
        self,
        backend: LocalAIBackend,
        model_group: ModelGroup,
    ) -> None:
        backend.setup()
        
        result = backend.sync_group(model_group, model_group.source_dir)
        
        assert result.success is True
        
        # Check YAML was created
        yaml_file = backend.models_dir / "model-test-model.yaml"
        assert yaml_file.exists()
        
        content = yaml_file.read_text()
        assert "name:" in content
        assert "backend:" in content
    
    def test_sync_group_with_mmproj_includes_vision(
        self,
        backend: LocalAIBackend,
        temp_dir: Path,
    ) -> None:
        backend.setup()
        
        source_dir = temp_dir / "source"
        source_dir.mkdir()
        
        model_file = source_dir / "vision-model.gguf"
        mmproj_file = source_dir / "mmproj-vision-model.gguf"
        model_file.write_bytes(b"GGUF")
        mmproj_file.write_bytes(b"MMPROJ")
        
        group = ModelGroup(
            base_name="vision-model",
            files=[ModelInfo(path=model_file)],
            mmproj_file=ModelInfo(path=mmproj_file),
            source_dir=source_dir,
        )
        
        backend.sync_group(group, source_dir)
        
        yaml_file = backend.models_dir / "model-vision-model.yaml"
        content = yaml_file.read_text()
        assert "mmproj:" in content


class TestLMStudioBackend:
    """Integration tests for LMStudioBackend."""
    
    @pytest.fixture
    def backend(self, temp_dir: Path) -> LMStudioBackend:
        config = LMStudioConfig(output_dir=temp_dir / "lmstudio_models")
        return LMStudioBackend(config)
    
    @pytest.fixture
    def model_group(self, temp_dir: Path) -> ModelGroup:
        source_file = temp_dir / "source" / "model.gguf"
        source_file.parent.mkdir()
        source_file.write_bytes(b"GGUF content")
        
        return ModelGroup(
            base_name="test-model",
            files=[ModelInfo(path=source_file)],
            source_dir=source_file.parent,
        )
    
    def test_setup_creates_directories(self, backend: LMStudioBackend) -> None:
        backend.setup()
        assert backend.output_dir.exists()
        assert backend.manifest_dir.exists()
    
    def test_sync_group_creates_manifest(
        self,
        backend: LMStudioBackend,
        model_group: ModelGroup,
    ) -> None:
        backend.setup()
        
        result = backend.sync_group(model_group, model_group.source_dir)
        
        assert result.success is True
        
        # Check manifest was created
        manifest_file = backend.manifest_dir / "test-model.json"
        assert manifest_file.exists()
        
        import json
        manifest = json.loads(manifest_file.read_text())
        assert manifest["id"] == "test-model"
        assert manifest["has_vision"] is False
