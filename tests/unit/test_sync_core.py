"""Tests for SyncEngine and related classes."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from gguf_sync.core.models import AppConfig, LlamaCppConfig, ModelInfo, SyncEvent, SyncEventType
from gguf_sync.core.sync import ModelFilter, SyncEngine


class TestModelFilter:
    """Tests for ModelFilter class."""

    def test_init_empty(self) -> None:
        """Test ModelFilter initialization without patterns."""
        filter_obj = ModelFilter()
        assert filter_obj.patterns == []

    def test_init_with_patterns(self) -> None:
        """Test ModelFilter initialization with patterns."""
        filter_obj = ModelFilter(["model-a", "model-b"])
        assert filter_obj.patterns == ["model-a", "model-b"]

    def test_should_ignore_exact_match(self) -> None:
        """Test should_ignore with exact match."""
        filter_obj = ModelFilter(["model-a"])
        assert filter_obj.should_ignore("model-a") is True

    def test_should_ignore_no_match(self) -> None:
        """Test should_ignore with no match."""
        filter_obj = ModelFilter(["model-a"])
        assert filter_obj.should_ignore("model-b") is False

    def test_should_ignore_empty_patterns(self) -> None:
        """Test should_ignore with empty patterns."""
        filter_obj = ModelFilter()
        assert filter_obj.should_ignore("model-a") is False

    def test_should_ignore_wildcard(self) -> None:
        """Test should_ignore with wildcard pattern."""
        filter_obj = ModelFilter(["model-*"])
        assert filter_obj.should_ignore("model-a") is True
        assert filter_obj.should_ignore("model-b") is True
        assert filter_obj.should_ignore("other-model") is False


class TestSyncEngineInit:
    """Tests for SyncEngine initialization."""

    def test_init(self, tmp_path: Path) -> None:
        """Test SyncEngine initialization."""
        config = AppConfig(source_dir=tmp_path)
        backends: list = []

        engine = SyncEngine(config, backends)
        assert engine.config == config
        assert engine.backends == backends


class TestSyncEngineHelpers:
    """Tests for SyncEngine helper methods."""

    @pytest.fixture
    def engine(self, tmp_path: Path) -> SyncEngine:
        """Create a SyncEngine instance."""
        config = AppConfig(source_dir=tmp_path)
        backends: list = []
        return SyncEngine(config, backends)

    def test_get_context_size_no_metadata(self, engine: SyncEngine) -> None:
        """Test _get_context_size with no metadata."""
        backend = MagicMock()
        backend.config = MagicMock()
        backend.config.context_size = 4096

        result = engine._get_context_size(backend, None)
        assert result == 4096

    def test_get_context_size_with_metadata(self, engine: SyncEngine) -> None:
        """Test _get_context_size with metadata."""
        backend = MagicMock()
        backend.config = MagicMock()
        backend.config.context_size = None  # No backend override

        metadata = MagicMock()
        metadata.context_length = 8192

        result = engine._get_context_size(backend, metadata)
        assert result == 8192

    def test_get_gpu_layers_with_config(self, engine: SyncEngine) -> None:
        """Test _get_gpu_layers with config value."""
        backend = MagicMock()
        backend.config = MagicMock()
        backend.config.gpu_layers = 33

        result = engine._get_gpu_layers(backend)
        assert result == 33

    def test_get_gpu_layers_none(self, engine: SyncEngine) -> None:
        """Test _get_gpu_layers with None."""
        backend = MagicMock()
        backend.config = MagicMock()
        backend.config.gpu_layers = None
        # When gpu_layers is None, it defaults to -1 (use all layers)
        result = engine._get_gpu_layers(backend)
        assert result == -1

    def test_get_threads_with_config(self, engine: SyncEngine) -> None:
        """Test _get_threads with config value."""
        backend = MagicMock()
        backend.config = MagicMock()
        backend.config.threads = 4

        result = engine._get_threads(backend)
        assert result == 4

    def test_should_skip_backend_no_filter(self, engine: SyncEngine) -> None:
        """Test _should_skip_backend with no filter."""
        backend = MagicMock()
        backend.config = MagicMock()
        backend.config.filter = None

        result = engine._should_skip_backend(backend, "model-a")
        assert result is False

    def test_should_skip_backend_with_global_filter(self, engine: SyncEngine) -> None:
        """Test _should_skip_backend with global filter."""
        backend = MagicMock()
        backend.name = "test-backend"
        backend.output_dir = MagicMock()
        backend.output_dir.__truediv__ = MagicMock(
            return_value=MagicMock(exists=MagicMock(return_value=False))
        )
        backend.config = MagicMock()
        backend.config.ignore_file = None

        # Set up global filter to ignore model-a
        engine._global_filter = ModelFilter(["model-a"])

        # Should skip model-a due to global filter
        result = engine._should_skip_backend(backend, "model-a")
        assert result is True

        # Should not skip model-b
        result = engine._should_skip_backend(backend, "model-b")
        assert result is False

    def test_should_skip_backend_disabled_filter(self, engine: SyncEngine) -> None:
        """Test _should_skip_backend with disabled filter."""
        backend = MagicMock()
        backend.config = MagicMock()
        backend.config.filter = MagicMock()
        backend.config.filter.enabled = False
        backend.config.filter.patterns = ["model-a"]

        result = engine._should_skip_backend(backend, "model-a")
        assert result is False

    def test_backends_need_metadata_localai(self, engine: SyncEngine) -> None:
        """Test _backends_need_metadata for LocalAI backend."""
        from gguf_sync.backends.localai import LocalAIBackend

        backend = MagicMock(spec=LocalAIBackend)
        backend.name = "LocalAI"
        backend.localai_config = MagicMock()
        backend.localai_config.generate_yaml = True
        engine.backends = [backend]

        result = engine._backends_need_metadata()
        assert result is True

    def test_backends_need_metadata_lmstudio(self, engine: SyncEngine) -> None:
        """Test _backends_need_metadata for LMStudio backend."""
        from gguf_sync.backends.lmstudio import LMStudioBackend

        backend = MagicMock(spec=LMStudioBackend)
        backend.name = "LM Studio"
        backend.lmstudio_config = MagicMock()
        backend.lmstudio_config.generate_manifest = True
        engine.backends = [backend]

        result = engine._backends_need_metadata()
        assert result is True

    def test_backends_need_metadata_false(self, engine: SyncEngine) -> None:
        """Test _backends_need_metadata when backend doesn't need it."""
        backend = MagicMock()
        backend.name = "llama.cpp"
        engine.backends = [backend]

        result = engine._backends_need_metadata()
        assert result is False


class TestSyncEngineFileNeedsSync:
    """Tests for _file_needs_sync method."""

    @pytest.fixture
    def engine(self, tmp_path: Path) -> SyncEngine:
        """Create a SyncEngine instance."""
        config = AppConfig(source_dir=tmp_path)
        backends: list = []
        return SyncEngine(config, backends)

    def test_file_needs_sync_no_backends(self, engine: SyncEngine) -> None:
        """Test _file_needs_sync with no backends."""
        # With no backends, there's nothing to sync to
        model_info = MagicMock()
        model_info.mtime = 12345
        model_info.file_size = 1000
        model_info.path = MagicMock()
        model_info.path.stem = "model"
        model_info.path.name = "model.gguf"

        result = engine._file_needs_sync(model_info)
        assert result is False


class TestSyncEngineSetup:
    """Tests for setup method."""

    def test_setup_creates_directories(self, tmp_path: Path) -> None:
        """Test setup creates necessary directories."""
        config = AppConfig(source_dir=tmp_path)
        backend = MagicMock()
        backend.output_dir = tmp_path / "backend"
        backend.config = MagicMock()
        backend.config.enabled = True

        engine = SyncEngine(config, [backend])

        with patch.object(backend, "setup") as mock_setup:
            engine.setup()
            mock_setup.assert_called_once()


class TestSyncEngineStats:
    """Tests for get_stats method."""

    def test_get_stats(self, tmp_path: Path) -> None:
        """Test get_stats returns expected structure."""
        config = AppConfig(source_dir=tmp_path)
        engine = SyncEngine(config, [])

        stats = engine.get_stats()
        assert "source_dir" in stats
        assert "total_files" in stats
        assert "total_groups" in stats
        assert "backends" in stats
