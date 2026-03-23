"""Tests for CLI commands in main.py."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from gguf_sync.core.exceptions import GGUFSyncError


class TestCLIFunctionsDirectly:
    """Tests for CLI functions called directly (not through Typer)."""

    def test_get_backends_empty_config(self) -> None:
        """Test get_backends with empty config."""
        from gguf_sync.core.models import AppConfig
        from gguf_sync.main import get_backends

        config = AppConfig()
        result = get_backends(config)
        assert result == []

    def test_get_backends_all_disabled(self) -> None:
        """Test get_backends with all backends disabled."""
        from gguf_sync.core.models import AppConfig, LlamaCppConfig
        from gguf_sync.main import get_backends

        config = AppConfig(
            source_dir=Path("/models"),
            backends={"llama_cpp": LlamaCppConfig(output_dir=Path("/tmp"), enabled=False)},
        )
        result = get_backends(config)
        assert result == []

    def test_version_callback_true(self) -> None:
        """Test version_callback with True raises Exit."""
        import typer

        from gguf_sync.main import version_callback

        with pytest.raises(typer.Exit):
            version_callback(True)

    def test_version_callback_false(self) -> None:
        """Test version_callback with False does not raise."""
        from gguf_sync.main import version_callback

        # Should not raise
        version_callback(False)


class TestMainCallback:
    """Tests for main callback function."""

    @patch("gguf_sync.main.setup_logging")
    def test_main_callback_sets_up_logging(self, mock_setup: MagicMock) -> None:
        """Test that main callback sets up logging."""
        from gguf_sync.main import main

        # Call main callback with no options
        main(version=False, config_file=None, verbose=False, json_logs=False)

        mock_setup.assert_called_once_with(verbose=False, json_format=False)

    @patch("gguf_sync.main.setup_logging")
    def test_main_callback_verbose_logging(self, mock_setup: MagicMock) -> None:
        """Test that main callback sets up verbose logging."""
        from gguf_sync.main import main

        main(version=False, config_file=None, verbose=True, json_logs=False)

        mock_setup.assert_called_once_with(verbose=True, json_format=False)

    @patch("gguf_sync.main.setup_logging")
    def test_main_callback_json_logging(self, mock_setup: MagicMock) -> None:
        """Test that main callback sets up JSON logging."""
        from gguf_sync.main import main

        main(version=False, config_file=None, verbose=False, json_logs=True)

        mock_setup.assert_called_once_with(verbose=False, json_format=True)


class TestSyncCommandDirectly:
    """Tests for sync function called directly."""

    @patch("gguf_sync.main.ConfigLoader")
    @patch("gguf_sync.main.get_backends")
    def test_sync_no_backends_raises_error(
        self, mock_get_backends: MagicMock, mock_loader: MagicMock
    ) -> None:
        """Test sync raises error when no backends enabled."""
        import typer

        from gguf_sync.core.models import AppConfig
        from gguf_sync.main import sync

        mock_config = AppConfig()
        mock_loader.return_value.load.return_value = mock_config
        mock_get_backends.return_value = []

        with pytest.raises(typer.Exit) as exc_info:
            sync()
        assert exc_info.value.exit_code == 1


class TestWatchCommandDirectly:
    """Tests for watch function called directly."""

    @patch("gguf_sync.main.ConfigLoader")
    @patch("gguf_sync.main.get_backends")
    def test_watch_no_backends_raises_error(
        self, mock_get_backends: MagicMock, mock_loader: MagicMock
    ) -> None:
        """Test watch raises error when no backends enabled."""
        import typer

        from gguf_sync.core.models import AppConfig
        from gguf_sync.main import watch

        mock_config = AppConfig()
        mock_loader.return_value.load.return_value = mock_config
        mock_get_backends.return_value = []

        with pytest.raises(typer.Exit) as exc_info:
            watch()
        assert exc_info.value.exit_code == 1


class TestConfigCommandDirectly:
    """Tests for config function called directly."""

    @patch("gguf_sync.main.ConfigLoader")
    @patch("builtins.open")
    def test_config_generate_writes_file(
        self, mock_open: MagicMock, mock_loader: MagicMock
    ) -> None:
        """Test config generate writes configuration file."""
        from gguf_sync.main import config

        mock_loader.return_value.generate_default_config.return_value = "test: config"

        config(generate=True, output=Path("/tmp/config.yaml"))

        mock_open.assert_called_once()

    @patch("gguf_sync.main.ConfigLoader")
    @patch("gguf_sync.main.console")
    def test_config_show_displays_current(
        self, mock_console: MagicMock, mock_loader: MagicMock
    ) -> None:
        """Test config show displays current configuration."""
        from gguf_sync.core.models import AppConfig
        from gguf_sync.main import config

        mock_config = AppConfig()
        mock_loader.return_value.load.return_value = mock_config

        config(generate=False, output=Path("gguf_sync.yaml"))

        mock_console.print.assert_called()


class TestDiscoverCommandDirectly:
    """Tests for discover function called directly."""

    @patch("gguf_sync.main.BackendDiscovery")
    @patch("gguf_sync.main.console")
    def test_discover_no_backends_shows_message(
        self, mock_console: MagicMock, mock_discovery: MagicMock
    ) -> None:
        """Test discover shows message when no backends found."""
        from gguf_sync.main import discover

        mock_discovery.return_value.discover_all.return_value = []

        discover(generate_config=False, output=Path("gguf_sync.yaml"))

        mock_console.print.assert_called_with("[yellow]No backends discovered[/yellow]")

    @patch("gguf_sync.main.BackendDiscovery")
    @patch("gguf_sync.main.console")
    def test_discover_with_backends_shows_table(
        self, mock_console: MagicMock, mock_discovery: MagicMock
    ) -> None:
        """Test discover shows table when backends found."""
        from gguf_sync.main import discover

        mock_backend = MagicMock()
        mock_backend.name = "test-backend"
        mock_backend.backend_type = "test"
        mock_backend.install_dir = Path("/test")
        mock_backend.models_dir = None
        mock_backend.is_running = False
        mock_backend.port = None
        mock_discovery.return_value.discover_all.return_value = [mock_backend]

        discover(generate_config=False, output=Path("gguf_sync.yaml"))

        # Should print a table
        assert mock_console.print.call_count >= 1


class TestServiceCommandDirectly:
    """Tests for service function called directly."""

    @patch("gguf_sync.main.ServiceInstaller")
    @patch("gguf_sync.main.console")
    def test_service_install(self, mock_console: MagicMock, mock_installer: MagicMock) -> None:
        """Test service install command."""
        from gguf_sync.main import service

        service(action="install", name="test-service")

        mock_installer.return_value.install.assert_called_once()
        mock_console.print.assert_called()

    @patch("gguf_sync.main.ServiceInstaller")
    @patch("gguf_sync.main.console")
    def test_service_uninstall(self, mock_console: MagicMock, mock_installer: MagicMock) -> None:
        """Test service uninstall command."""
        from gguf_sync.main import service

        service(action="uninstall", name="test-service")

        mock_installer.return_value.uninstall.assert_called_once()
        mock_console.print.assert_called()

    @patch("gguf_sync.main.ServiceInstaller")
    @patch("gguf_sync.main.console")
    def test_service_start(self, mock_console: MagicMock, mock_installer: MagicMock) -> None:
        """Test service start command."""
        from gguf_sync.main import service

        service(action="start", name="test-service")

        mock_installer.return_value.start.assert_called_once()
        mock_console.print.assert_called()

    @patch("gguf_sync.main.ServiceInstaller")
    @patch("gguf_sync.main.console")
    def test_service_stop(self, mock_console: MagicMock, mock_installer: MagicMock) -> None:
        """Test service stop command."""
        from gguf_sync.main import service

        service(action="stop", name="test-service")

        mock_installer.return_value.stop.assert_called_once()
        mock_console.print.assert_called()

    @patch("gguf_sync.main.ServiceInstaller")
    @patch("gguf_sync.main.console")
    def test_service_status_installed_active(
        self, mock_console: MagicMock, mock_installer: MagicMock
    ) -> None:
        """Test service status when installed and active."""
        from gguf_sync.main import service

        mock_installer.return_value.status.return_value = {
            "installed": True,
            "active": True,
        }

        service(action="status", name="test-service")

        mock_console.print.assert_called()

    @patch("gguf_sync.main.ServiceInstaller")
    @patch("gguf_sync.main.console")
    def test_service_status_not_installed(
        self, mock_console: MagicMock, mock_installer: MagicMock
    ) -> None:
        """Test service status when not installed."""
        from gguf_sync.main import service

        mock_installer.return_value.status.return_value = {"installed": False}

        service(action="status", name="test-service")

        mock_console.print.assert_called()

    def test_service_invalid_action_raises_error(self) -> None:
        """Test service with invalid action raises Exit."""
        import typer

        from gguf_sync.main import service

        with pytest.raises(typer.Exit) as exc_info:
            service(action="invalid", name="test-service")
        assert exc_info.value.exit_code == 1


class TestMainErrorHandling:
    """Tests for error handling in main functions."""

    @patch("gguf_sync.main.ConfigLoader")
    def test_sync_gguf_error(self, mock_loader: MagicMock) -> None:
        """Test sync handles GGUFSyncError."""
        import typer

        from gguf_sync.main import sync

        mock_loader.return_value.load.side_effect = GGUFSyncError("Test error")

        with pytest.raises(typer.Exit) as exc_info:
            sync()
        assert exc_info.value.exit_code == 1

    @patch("gguf_sync.main.ConfigLoader")
    def test_sync_generic_error(self, mock_loader: MagicMock) -> None:
        """Test sync handles generic exceptions."""
        import typer

        from gguf_sync.main import sync

        mock_loader.return_value.load.side_effect = Exception("Generic error")

        with pytest.raises(typer.Exit) as exc_info:
            sync()
        assert exc_info.value.exit_code == 1

    @patch("gguf_sync.main.BackendDiscovery")
    def test_discover_generic_error(self, mock_discovery: MagicMock) -> None:
        """Test discover handles generic exceptions."""
        import typer

        from gguf_sync.main import discover

        mock_discovery.return_value.discover_all.side_effect = Exception("Test error")

        with pytest.raises(typer.Exit) as exc_info:
            discover(generate_config=False, output=Path("test.yaml"))
        assert exc_info.value.exit_code == 1
