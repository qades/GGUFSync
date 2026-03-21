"""Integration tests for SyncEngine."""

from __future__ import annotations

from pathlib import Path

import pytest

from link_models.backends import LlamaCppBackend, LocalAIBackend
from link_models.core.models import (
    AppConfig,
    LlamaCppConfig,
    LocalAIConfig,
    SyncEvent,
    SyncEventType,
)
from link_models.core.sync import SyncEngine


class TestSyncEngine:
    """Integration tests for SyncEngine."""
    
    @pytest.fixture
    def setup_dirs(self, temp_dir: Path) -> dict[str, Path]:
        """Create source and destination directories."""
        dirs = {
            "source": temp_dir / "models",
            "llama": temp_dir / "llama_models",
            "localai": temp_dir / "localai_models",
        }
        for d in dirs.values():
            d.mkdir()
        return dirs
    
    @pytest.fixture
    def engine(self, setup_dirs: dict[str, Path]) -> SyncEngine:
        """Create a configured SyncEngine."""
        config = AppConfig(
            source_dir=setup_dirs["source"],
            backends={
                "llama_cpp": LlamaCppConfig(output_dir=setup_dirs["llama"]),
                "localai": LocalAIConfig(output_dir=setup_dirs["localai"]),
            }
        )
        
        backends = [
            LlamaCppBackend(config.backends["llama_cpp"]),
            LocalAIBackend(config.backends["localai"]),
        ]
        
        return SyncEngine(config, backends)
    
    def test_setup_creates_directories(self, engine: SyncEngine) -> None:
        engine.setup()
        
        for backend in engine.backends:
            assert backend.output_dir.exists()
    
    def test_full_sync_single_model(
        self,
        engine: SyncEngine,
        setup_dirs: dict[str, Path],
    ) -> None:
        # Create a model file
        model_file = setup_dirs["source"] / "test-model.gguf"
        model_file.write_bytes(b"GGUF" + b"\x00" * 100)
        
        engine.setup()
        results = engine.full_sync()
        
        assert "llama.cpp" in results
        assert "LocalAI" in results
        
        # Check files were created
        assert (setup_dirs["llama"] / "test-model" / "test-model.gguf").exists()
        assert (setup_dirs["localai"] / "test-model" / "test-model.gguf").exists()
        assert (setup_dirs["localai"] / "model-test-model.yaml").exists()
    
    def test_full_sync_multipart_model(
        self,
        engine: SyncEngine,
        setup_dirs: dict[str, Path],
    ) -> None:
        # Create multipart model files
        (setup_dirs["source"] / "big-model-00001-of-00002.gguf").write_bytes(b"GGUF1" + b"\x00" * 100)
        (setup_dirs["source"] / "big-model-00002-of-00002.gguf").write_bytes(b"GGUF2" + b"\x00" * 100)
        
        engine.setup()
        results = engine.full_sync()
        
        # Check all parts were linked
        llama_dir = setup_dirs["llama"] / "big-model"
        assert (llama_dir / "big-model-00001-of-00002.gguf").exists()
        assert (llama_dir / "big-model-00002-of-00002.gguf").exists()
    
    def test_full_sync_with_mmproj(
        self,
        engine: SyncEngine,
        setup_dirs: dict[str, Path],
    ) -> None:
        # Create model with mmproj
        (setup_dirs["source"] / "vision-model.gguf").write_bytes(b"GGUF" + b"\x00" * 100)
        (setup_dirs["source"] / "mmproj-vision-model.gguf").write_bytes(b"MMPROJ" + b"\x00" * 100)
        
        engine.setup()
        engine.full_sync()
        
        # Check mmproj was linked
        llama_dir = setup_dirs["llama"] / "vision-model"
        assert (llama_dir / "mmproj-vision-model.gguf").exists()
    
    def test_handle_event_file_created(
        self,
        engine: SyncEngine,
        setup_dirs: dict[str, Path],
    ) -> None:
        engine.setup()
        
        # Initial sync
        engine.full_sync()
        
        # Create new file
        new_file = setup_dirs["source"] / "new-model.gguf"
        new_file.write_bytes(b"GGUF new" + b"\x00" * 100)
        
        # Simulate event
        event = SyncEvent(
            event_type=SyncEventType.FILE_CREATED,
            path=new_file,
            source_dir=setup_dirs["source"],
        )
        
        results = engine.handle_event(event)
        
        # Check file was synced
        assert (setup_dirs["llama"] / "new-model" / "new-model.gguf").exists()
    
    def test_handle_event_file_deleted_from_source(
        self,
        engine: SyncEngine,
        setup_dirs: dict[str, Path],
    ) -> None:
        # Create and sync file
        model_file = setup_dirs["source"] / "delete-me.gguf"
        model_file.write_bytes(b"GGUF" + b"\x00" * 100)
        
        engine.setup()
        engine.full_sync()
        
        # Verify file exists
        assert (setup_dirs["llama"] / "delete-me" / "delete-me.gguf").exists()
        
        # Delete file
        model_file.unlink()
        
        # Simulate deletion event
        event = SyncEvent(
            event_type=SyncEventType.FILE_DELETED,
            path=model_file,
            source_dir=setup_dirs["source"],
        )
        
        results = engine.handle_event(event)
        
        # Check file was removed from backends
        assert not (setup_dirs["llama"] / "delete-me").exists()
    
    def test_cleanup_orphans(
        self,
        engine: SyncEngine,
        setup_dirs: dict[str, Path],
    ) -> None:
        engine.setup()
        engine.full_sync()
        
        # Create orphan directory
        orphan_dir = setup_dirs["llama"] / "orphan-model"
        orphan_dir.mkdir()
        (orphan_dir / "file.gguf").write_text("orphan")
        
        # Create orphan YAML
        orphan_yaml = setup_dirs["localai"] / "model-orphan.yaml"
        orphan_yaml.write_text("orphan yaml")
        
        # Add a model to source to trigger cleanup
        (setup_dirs["source"] / "real-model.gguf").write_bytes(b"GGUF" + b"\x00" * 100)
        engine.full_sync()
        
        # Orphans should be cleaned up
        assert not orphan_dir.exists()
        assert not orphan_yaml.exists()
    
    def test_stats_reporting(
        self,
        engine: SyncEngine,
        setup_dirs: dict[str, Path],
    ) -> None:
        (setup_dirs["source"] / "model1.gguf").write_bytes(b"GGUF1" + b"\x00" * 100)
        (setup_dirs["source"] / "model2.gguf").write_bytes(b"GGUF2" + b"\x00" * 100)
        
        engine.setup()
        engine.full_sync()
        
        stats = engine.get_stats()
        
        assert stats["source_dir"] == str(setup_dirs["source"])
        assert stats["total_files"] == 2
        assert stats["total_groups"] == 2
        assert "llama.cpp" in stats["backends"]
        assert "LocalAI" in stats["backends"]
