"""Cross-platform service installation and management."""

from __future__ import annotations

import os
import platform
import re
import shutil
import stat
import subprocess
import sys
from pathlib import Path
from typing import Any

from .constants import DEFAULT_SERVICE_NAME
from .exceptions import ServiceError
from .logging import get_logger

logger = get_logger(__name__)

# Valid service name pattern (alphanumeric, hyphen, underscore)
VALID_SERVICE_NAME_PATTERN = re.compile(r'^[a-zA-Z0-9_-]+$')


class ServiceInstaller:
    """Cross-platform service installer."""
    
    def __init__(
        self,
        service_name: str = DEFAULT_SERVICE_NAME,
        executable_path: Path | None = None,
    ) -> None:
        """Initialize service installer.
        
        Args:
            service_name: Name of the service
            executable_path: Path to the executable (defaults to current process)
            
        Raises:
            ServiceError: If service_name contains invalid characters
        """
        # Validate service name to prevent command injection
        if not VALID_SERVICE_NAME_PATTERN.match(service_name):
            raise ServiceError(
                f"Invalid service name: {service_name}. "
                "Only alphanumeric characters, hyphens, and underscores are allowed."
            )
        self.service_name = service_name
        self.executable_path = executable_path or Path(sys.executable)
        self.system = platform.system()
    
    def install(self, args: list[str] | None = None) -> None:
        """Install the service.
        
        Args:
            args: Additional arguments to pass to the service
            
        Raises:
            ServiceError: If installation fails
        """
        if self.system == "Linux":
            self._install_systemd(args)
        elif self.system == "Darwin":
            self._install_launchd(args)
        elif self.system == "Windows":
            self._install_windows_service(args)
        else:
            raise ServiceError(f"Unsupported platform: {self.system}")
    
    def uninstall(self) -> None:
        """Uninstall the service.
        
        Raises:
            ServiceError: If uninstallation fails
        """
        if self.system == "Linux":
            self._uninstall_systemd()
        elif self.system == "Darwin":
            self._uninstall_launchd()
        elif self.system == "Windows":
            self._uninstall_windows_service()
        else:
            raise ServiceError(f"Unsupported platform: {self.system}")
    
    def start(self) -> None:
        """Start the service.
        
        Raises:
            ServiceError: If the service fails to start
        """
        try:
            if self.system == "Linux":
                subprocess.run(
                    ["systemctl", "start", self.service_name],
                    check=True,
                    capture_output=True,
                    text=True,
                )
            elif self.system == "Darwin":
                subprocess.run(
                    ["launchctl", "start", self.service_name],
                    check=True,
                    capture_output=True,
                    text=True,
                )
            elif self.system == "Windows":
                subprocess.run(
                    ["sc", "start", self.service_name],
                    check=True,
                    capture_output=True,
                    text=True,
                )
            else:
                raise ServiceError(f"Unsupported platform: {self.system}")
        except subprocess.CalledProcessError as e:
            error_msg = e.stderr if e.stderr else str(e)
            raise ServiceError(f"Failed to start service: {error_msg}") from e
        except FileNotFoundError as e:
            raise ServiceError(f"Service management command not found: {e}") from e
    
    def stop(self) -> None:
        """Stop the service.
        
        Raises:
            ServiceError: If the service fails to stop
        """
        try:
            if self.system == "Linux":
                subprocess.run(
                    ["systemctl", "stop", self.service_name],
                    check=True,
                    capture_output=True,
                    text=True,
                )
            elif self.system == "Darwin":
                subprocess.run(
                    ["launchctl", "stop", self.service_name],
                    check=True,
                    capture_output=True,
                    text=True,
                )
            elif self.system == "Windows":
                subprocess.run(
                    ["sc", "stop", self.service_name],
                    check=True,
                    capture_output=True,
                    text=True,
                )
            else:
                raise ServiceError(f"Unsupported platform: {self.system}")
        except subprocess.CalledProcessError as e:
            error_msg = e.stderr if e.stderr else str(e)
            raise ServiceError(f"Failed to stop service: {error_msg}") from e
        except FileNotFoundError as e:
            raise ServiceError(f"Service management command not found: {e}") from e
    
    def status(self) -> dict[str, Any]:
        """Get service status.
        
        Returns:
            Dictionary with status information
        """
        try:
            if self.system == "Linux":
                result = subprocess.run(
                    ["systemctl", "status", self.service_name],
                    capture_output=True,
                    text=True,
                )
                return {
                    "installed": result.returncode in (0, 3),
                    "active": "Active: active" in result.stdout,
                    "output": result.stdout,
                }
            elif self.system == "Darwin":
                result = subprocess.run(
                    ["launchctl", "list", self.service_name],
                    capture_output=True,
                    text=True,
                )
                return {
                    "installed": result.returncode == 0,
                    "active": "" if result.returncode != 0 else "running",
                    "output": result.stdout,
                }
            elif self.system == "Windows":
                result = subprocess.run(
                    ["sc", "query", self.service_name],
                    capture_output=True,
                    text=True,
                )
                return {
                    "installed": result.returncode == 0,
                    "active": "RUNNING" in result.stdout,
                    "output": result.stdout,
                }
        except Exception as e:
            return {"installed": False, "error": str(e)}
        
        # This should not be reached, but just in case
        return {"installed": False, "error": f"Unknown platform: {self.system}"}
    
    def _install_systemd(self, args: list[str] | None = None) -> None:
        """Install systemd service on Linux."""
        # Create service user if it doesn't exist
        try:
            subprocess.run(
                ["id", "-u", "localai"],
                capture_output=True,
                check=True,
            )
        except subprocess.CalledProcessError:
            logger.info("Creating service user 'localai'")
            subprocess.run(
                [
                    "useradd",
                    "-r",
                    "-s", "/bin/false",
                    "-d", "/nonexistent",
                    "-M",
                    "localai",
                ],
                check=False,
            )
        
        # Determine executable path
        if getattr(sys, 'frozen', False):
            # Running as compiled executable
            exec_path = self.executable_path
        else:
            # Running as Python script
            exec_path = self.executable_path
            args = ["-m", "link_models", "watch"] + (args or [])
        
        # Create systemd service file
        service_content = f"""[Unit]
Description=Link Models Watcher
After=network.target

[Service]
Type=simple
ExecStart={exec_path} {' '.join(args or ['watch'])}
Restart=always
RestartSec=10
User=localai
Group=localai

# Security hardening
NoNewPrivileges=true
PrivateTmp=true
ProtectHome=true
ProtectSystem=strict
ReadWritePaths=/llama_models /localai_models
ReadOnlyPaths=/models

[Install]
WantedBy=multi-user.target
"""
        
        service_path = Path(f"/etc/systemd/system/{self.service_name}.service")
        
        try:
            # Write service file with restricted permissions (readable only by root)
            with open(service_path, "w") as f:
                f.write(service_content)
            os.chmod(service_path, stat.S_IRUSR | stat.S_IWUSR)  # 0o600
            
            # Reload systemd
            subprocess.run(["systemctl", "daemon-reload"], check=True)
            logger.info(f"Installed systemd service: {service_path}")
            
        except PermissionError:
            raise ServiceError(
                "Permission denied. Run with sudo to install service."
            )
        except Exception as e:
            raise ServiceError(f"Failed to install systemd service: {e}")
    
    def _uninstall_systemd(self) -> None:
        """Uninstall systemd service."""
        service_path = Path(f"/etc/systemd/system/{self.service_name}.service")
        
        try:
            # Stop and disable service
            subprocess.run(
                ["systemctl", "stop", self.service_name],
                capture_output=True,
            )
            subprocess.run(
                ["systemctl", "disable", self.service_name],
                capture_output=True,
            )
            
            # Remove service file
            if service_path.exists():
                service_path.unlink()
            
            # Reload systemd
            subprocess.run(["systemctl", "daemon-reload"], check=True)
            logger.info(f"Uninstalled systemd service: {service_path}")
            
        except PermissionError:
            raise ServiceError(
                "Permission denied. Run with sudo to uninstall service."
            )
        except Exception as e:
            raise ServiceError(f"Failed to uninstall systemd service: {e}")
    
    def _install_launchd(self, args: list[str] | None = None) -> None:
        """Install launchd plist on macOS."""
        # Determine executable path
        if getattr(sys, 'frozen', False):
            exec_path = self.executable_path
        else:
            exec_path = self.executable_path
            args = ["-m", "link_models", "watch"] + (args or [])
        
        plist_content = f"""<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
    <key>Label</key>
    <string>{self.service_name}</string>
    <key>ProgramArguments</key>
    <array>
        <string>{exec_path}</string>
{chr(10).join(f'        <string>{arg}</string>' for arg in (args or ["watch"]))}
    </array>
    <key>RunAtLoad</key>
    <true/>
    <key>KeepAlive</key>
    <true/>
    <key>StandardOutPath</key>
    <string>{log_dir}/{self.service_name}.log</string>
    <key>StandardErrorPath</key>
    <string>{log_dir}/{self.service_name}.err</string>
</dict>
</plist>
"""
        
        plist_path = Path(f"~/Library/LaunchAgents/{self.service_name}.plist").expanduser()
        plist_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Use user log directory instead of system directory for permissions
        log_dir = Path(f"~/Library/Logs/{self.service_name}").expanduser()
        log_dir.mkdir(parents=True, exist_ok=True)
        
        try:
            with open(plist_path, "w") as f:
                f.write(plist_content)
            
            # Load the plist
            subprocess.run(["launchctl", "load", str(plist_path)], check=True)
            logger.info(f"Installed launchd plist: {plist_path}")
            
        except Exception as e:
            raise ServiceError(f"Failed to install launchd plist: {e}")
    
    def _uninstall_launchd(self) -> None:
        """Uninstall launchd plist."""
        plist_path = Path(f"~/Library/LaunchAgents/{self.service_name}.plist").expanduser()
        
        try:
            # Unload the plist
            subprocess.run(
                ["launchctl", "unload", str(plist_path)],
                capture_output=True,
            )
            
            # Remove plist file
            if plist_path.exists():
                plist_path.unlink()
            
            logger.info(f"Uninstalled launchd plist: {plist_path}")
            
        except Exception as e:
            raise ServiceError(f"Failed to uninstall launchd plist: {e}")
    
    def _install_windows_service(self, args: list[str] | None = None) -> None:
        """Install Windows service using sc or nssm."""
        # For Windows, we recommend using NSSM (Non-Sucking Service Manager)
        # or manual setup as Windows services require special handling
        raise ServiceError(
            "Windows service installation requires manual setup. "
            "Consider using NSSM (Non-Sucking Service Manager) or pywin32. "
            "Example with NSSM: nssm install link-models <path_to_executable>"
        )
    
    def _uninstall_windows_service(self) -> None:
        """Uninstall Windows service."""
        # Same limitation as install - Windows services require special handling
        raise ServiceError(
            "Windows service uninstallation requires manual setup. "
            "Consider using NSSM or manual removal. "
            "Example: nssm remove link-models confirm"
        )
