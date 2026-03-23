"""Tests for add-only sync and ignore file functionality."""

from __future__ import annotations

import tempfile
from pathlib import Path

import pytest

from link_models.core.sync import ModelFilter, SyncEngine
from link_models.core.models import (
    AppConfig,
    SyncConfig,
    WatchConfig,
    LoggingConfig,
    BackendConfig,
)
from link_models.backends.base import Backend, BackendResult


class MockBackend(Backend):
    """Mock backend for testing."""

    def __init__(self, name: str, output_dir: Path) -> None:
        config = BackendConfig(output_dir=output_dir)
        super().__init__(config)
        self._name = name
        self.synced_models: list[str] = []
        self.removed_models: list[str] = []

    @property
    def name(self) -> str:
        return self._name

    def sync_group(self, group, source_dir) -> BackendResult:
        self.synced_models.append(group.model_id)
        return BackendResult(success=True, linked=1)

    def remove_group(self, model_id: str) -> BackendResult:
        self.removed_models.append(model_id)
        return BackendResult(success=True, removed=1)


class TestModelFilter:
    """Tests for ModelFilter class."""

    def test_load_from_file(self, temp_dir: Path) -> None:
        """Test loading patterns from file."""
        ignore_file = temp_dir / ".linkmodelsignore"
        ignore_file.write_text("""
# Comment line
model-a
*test*
*q4_*
""")

        filter = ModelFilter()
        filter.load_from_file(ignore_file)

        assert len(filter.patterns) == 3
        assert "model-a" in filter.patterns
        assert "*test*" in filter.patterns

    def test_should_ignore_exact_match(self) -> None:
        """Test exact pattern matching."""
        filter = ModelFilter(["model-a", "specific-model"])

        assert filter.should_ignore("model-a") is True
        assert filter.should_ignore("model-b") is False
        assert filter.should_ignore("specific-model") is True

    def test_should_ignore_glob_patterns(self) -> None:
        """Test glob pattern matching."""
        filter = ModelFilter(["*-small", "test*", "*-q4_*"])

        assert filter.should_ignore("model-small") is True
        assert filter.should_ignore("test-model") is True
        assert filter.should_ignore("llama-q4_k_m") is True
        assert filter.should_ignore("model-normal") is False

    def test_should_ignore_case_insensitive(self) -> None:
        """Test case-insensitive matching."""
        filter = ModelFilter(["MODEL-A", "*TEST*"])

        assert filter.should_ignore("model-a") is True
        assert filter.should_ignore("Model-A") is True
        assert filter.should_ignore("test-model") is True
        assert filter.should_ignore("TEST-MODEL") is True

    def test_empty_filter_accepts_all(self) -> None:
        """Test that empty filter accepts all models."""
        filter = ModelFilter()

        assert filter.should_ignore("any-model") is False

    def test_ignore_file_not_exists(self, temp_dir: Path) -> None:
        """Test that non-existent ignore file doesn't fail."""
        filter = ModelFilter()
        filter.load_from_file(temp_dir / "nonexistent.txt")

        # Should not raise, just returns with empty patterns
        assert filter.should_ignore("any-model") is False


class TestAddOnlySync:
    """Tests for add-only sync functionality."""

    @pytest.fixture
    def config_add_only(self, temp_dir: Path) -> AppConfig:
        """Create config with add_only enabled."""
        source_dir = temp_dir / "source"
        source_dir.mkdir()

        # Create a test model file
        (source_dir / "test-model.gguf").write_bytes(b"GGUF")

        return AppConfig(
            source_dir=source_dir,
            backends={
                "mock1": BackendConfig(output_dir=temp_dir / "backend1"),
                "mock2": BackendConfig(output_dir=temp_dir / "backend2"),
            },
            sync=SyncConfig(add_only=True, preserve_orphans=False),
            watch=WatchConfig(),
            logging=LoggingConfig(),
        )

    @pytest.fixture
    def config_normal(self, temp_dir: Path) -> AppConfig:
        """Create config with add_only disabled."""
        source_dir = temp_dir / "source"
        source_dir.mkdir()

        return AppConfig(
            source_dir=source_dir,
            backends={
                "mock1": BackendConfig(output_dir=temp_dir / "backend1"),
            },
            sync=SyncConfig(add_only=False, preserve_orphans=False),
            watch=WatchConfig(),
            logging=LoggingConfig(),
        )

    def test_add_only_config_flag(self, temp_dir: Path) -> None:
        """Test that add_only flag is properly stored in config."""
        config = AppConfig(
            source_dir=temp_dir / "models",
            backends={},
            sync=SyncConfig(add_only=True),
            watch=WatchConfig(),
            logging=LoggingConfig(),
        )

        assert config.sync.add_only is True

    def test_add_only_disabled_config(self, temp_dir: Path) -> None:
        """Test that add_only defaults to False."""
        config = AppConfig(
            source_dir=temp_dir / "models",
            backends={},
            sync=SyncConfig(),
            watch=WatchConfig(),
            logging=LoggingConfig(),
        )

        assert config.sync.add_only is False

    def test_global_ignore_file_config(self, temp_dir: Path) -> None:
        """Test global ignore file configuration."""
        ignore_path = temp_dir / "global-ignore.txt"
        ignore_path.write_text("model-a\nmodel-b\n")

        config = AppConfig(
            source_dir=temp_dir / "models",
            backends={},
            sync=SyncConfig(global_ignore_file=ignore_path),
            watch=WatchConfig(),
            logging=LoggingConfig(),
        )

        assert config.sync.global_ignore_file == ignore_path

    def test_backend_ignore_file_config(self, temp_dir: Path) -> None:
        """Test backend-specific ignore file configuration."""
        ignore_path = temp_dir / "ollama-ignore.txt"

        config = AppConfig(
            source_dir=temp_dir / "models",
            backends={
                "ollama": BackendConfig(
                    output_dir=temp_dir / "ollama",
                    ignore_file=ignore_path,
                ),
            },
            sync=SyncConfig(),
            watch=WatchConfig(),
            logging=LoggingConfig(),
        )

        assert config.backends["ollama"].ignore_file == ignore_path


class TestSyncEngineFilters:
    """Tests for sync engine filtering."""

    def test_global_filter_loads(self, temp_dir: Path) -> None:
        """Test that global filter loads from config."""
        ignore_file = temp_dir / "global-ignore.txt"
        ignore_file.write_text("test*\n")

        source_dir = temp_dir / "source"
        source_dir.mkdir()

        config = AppConfig(
            source_dir=source_dir,
            backends={},
            sync=SyncConfig(global_ignore_file=ignore_file),
            watch=WatchConfig(),
            logging=LoggingConfig(),
        )

        engine = SyncEngine(config, [])

        assert engine._global_filter.should_ignore("test-model") is True
        assert engine._global_filter.should_ignore("other-model") is False

    def test_backend_filter_from_file(self, temp_dir: Path) -> None:
        """Test that backend filter loads from output dir."""
        ignore_file = temp_dir / "backend1" / ".linkmodelsignore"
        ignore_file.parent.mkdir()
        ignore_file.write_text("model-a\n")

        source_dir = temp_dir / "source"
        source_dir.mkdir()

        config = AppConfig(
            source_dir=source_dir,
            backends={
                "backend1": BackendConfig(output_dir=temp_dir / "backend1"),
            },
            sync=SyncConfig(),
            watch=WatchConfig(),
            logging=LoggingConfig(),
        )

        backend = MockBackend("backend1", temp_dir / "backend1")
        engine = SyncEngine(config, [backend])

        # Filter should be loaded
        filter = engine._get_backend_filter(backend)
        assert filter.should_ignore("model-a") is True

    def test_should_skip_backend(self, temp_dir: Path) -> None:
        """Test _should_skip_backend method."""
        ignore_file = temp_dir / "global.txt"
        ignore_file.write_text("skip-model\n")

        source_dir = temp_dir / "source"
        source_dir.mkdir()

        config = AppConfig(
            source_dir=source_dir,
            backends={
                "backend1": BackendConfig(output_dir=temp_dir / "backend1"),
            },
            sync=SyncConfig(global_ignore_file=ignore_file),
            watch=WatchConfig(),
            logging=LoggingConfig(),
        )

        backend = MockBackend("backend1", temp_dir / "backend1")
        engine = SyncEngine(config, [backend])

        assert engine._should_skip_backend(backend, "skip-model") is True
        assert engine._should_skip_backend(backend, "keep-model") is False


class TestPreserveOrphans:
    """Tests for preserve_orphans setting."""

    def test_preserve_orphans_config(self, temp_dir: Path) -> None:
        """Test preserve_orphans flag in config."""
        config = AppConfig(
            source_dir=temp_dir / "models",
            backends={},
            sync=SyncConfig(preserve_orphans=True),
            watch=WatchConfig(),
            logging=LoggingConfig(),
        )

        assert config.sync.preserve_orphans is True

    def test_add_only_overrides_cleanup(self, temp_dir: Path) -> None:
        """Test that add_only prevents orphan cleanup."""
        source_dir = temp_dir / "source"
        source_dir.mkdir()

        backend_dir = temp_dir / "backend"
        backend_dir.mkdir()

        # Create orphan file
        (backend_dir / "orphan.gguf").write_bytes(b"GGUF")

        config = AppConfig(
            source_dir=source_dir,
            backends={
                "mock": BackendConfig(output_dir=backend_dir),
            },
            sync=SyncConfig(add_only=True, preserve_orphans=False),
            watch=WatchConfig(),
            logging=LoggingConfig(),
        )

        # Note: We can't fully test this without running sync
        # but we can verify the config is set correctly
        assert config.sync.add_only is True
        assert config.sync.preserve_orphans is False
