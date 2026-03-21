#!/usr/bin/env python3
"""Build script for creating standalone executables."""

from __future__ import annotations

import argparse
import platform
import shutil
import subprocess
import sys
from pathlib import Path

# Project root
PROJECT_ROOT = Path(__file__).parent.parent.resolve()
SRC_DIR = PROJECT_ROOT / "src"
DIST_DIR = PROJECT_ROOT / "dist"
BUILD_DIR = PROJECT_ROOT / "build"


def get_version() -> str:
    """Get the package version."""
    sys.path.insert(0, str(SRC_DIR))
    from link_models import __version__
    return __version__


def clean_build_dirs() -> None:
    """Remove previous build artifacts."""
    print("Cleaning build directories...")
    for dir_path in [DIST_DIR, BUILD_DIR]:
        if dir_path.exists():
            shutil.rmtree(dir_path)
            print(f"  Removed {dir_path}")


def build_pyinstaller(onefile: bool = True) -> Path:
    """Build executable using PyInstaller.
    
    Args:
        onefile: Create a single file executable
        
    Returns:
        Path to the built executable
    """
    print("Building with PyInstaller...")
    
    # Ensure pyinstaller is installed
    try:
        import PyInstaller
    except ImportError:
        print("Error: PyInstaller not installed. Run: pip install pyinstaller")
        sys.exit(1)
    
    # Build command
    cmd = [
        sys.executable, "-m", "PyInstaller",
        "--name", f"link-models-{get_version()}",
        "--distpath", str(DIST_DIR),
        "--workpath", str(BUILD_DIR),
        "--specpath", str(BUILD_DIR),
        "--noconfirm",
    ]
    
    if onefile:
        cmd.append("--onefile")
    else:
        cmd.append("--onedir")
    
    # Add hidden imports
    hidden_imports = [
        "gguf",
        "gguf.gguf_reader",
        "yaml",
        "watchdog",
        "watchdog.observers",
        "pydantic",
        "rich",
        "rich.console",
        "rich.panel",
        "rich.table",
        "rich.text",
        "structlog",
        "typer",
        "anyio",
    ]
    
    for imp in hidden_imports:
        cmd.extend(["--hidden-import", imp])
    
    # Add data files
    cmd.extend(["--collect-all", "gguf"])
    
    # Entry point
    entry_point = SRC_DIR / "link_models" / "main.py"
    cmd.append(str(entry_point))
    
    # Run pyinstaller
    result = subprocess.run(cmd, capture_output=False)
    
    if result.returncode != 0:
        print("Error: PyInstaller build failed")
        sys.exit(1)
    
    # Determine output path
    system = platform.system().lower()
    ext = ".exe" if system == "windows" else ""
    
    if onefile:
        executable = DIST_DIR / f"link-models-{get_version()}{ext}"
    else:
        executable = DIST_DIR / f"link-models-{get_version()}" / f"link-models-{get_version()}{ext}"
    
    print(f"Built: {executable}")
    return executable


def build_nuitka() -> Path:
    """Build executable using Nuitka.
    
    Returns:
        Path to the built executable
    """
    print("Building with Nuitka...")
    
    try:
        import nuitka
    except ImportError:
        print("Error: Nuitka not installed. Run: pip install nuitka")
        sys.exit(1)
    
    entry_point = SRC_DIR / "link_models" / "main.py"
    
    cmd = [
        sys.executable, "-m", "nuitka",
        "--standalone",
        "--onefile",
        "--enable-plugin=upx" if shutil.which("upx") else "",
        "--lto=yes",
        "--assume-yes-for-downloads",
        "--output-dir", str(DIST_DIR),
        "--output-filename", f"link-models-{get_version()}",
        "--include-package", "gguf",
        "--include-package", "yaml",
        "--include-package", "watchdog",
        "--include-package", "pydantic",
        "--include-package", "rich",
        "--include-package", "structlog",
        "--include-package", "typer",
        "--include-package", "anyio",
        str(entry_point),
    ]
    
    # Remove empty strings
    cmd = [c for c in cmd if c]
    
    result = subprocess.run(cmd, capture_output=False)
    
    if result.returncode != 0:
        print("Error: Nuitka build failed")
        sys.exit(1)
    
    # Determine output path
    system = platform.system().lower()
    ext = ".exe" if system == "windows" else ""
    executable = DIST_DIR / f"link-models-{get_version()}{ext}"
    
    print(f"Built: {executable}")
    return executable


def create_installer(executable: Path) -> None:
    """Create platform-specific installer.
    
    Args:
        executable: Path to the built executable
    """
    print("Creating installer...")
    
    system = platform.system()
    
    if system == "Linux":
        # Create a simple tarball with install script
        install_script = '''#!/bin/bash
set -e

echo "Installing link-models..."

# Determine install location
if [ "$EUID" -eq 0 ]; then
    INSTALL_DIR="/usr/local/bin"
else
    INSTALL_DIR="$HOME/.local/bin"
    mkdir -p "$INSTALL_DIR"
fi

# Copy executable
cp link-models-* "$INSTALL_DIR/link-models"
chmod +x "$INSTALL_DIR/link-models"

echo "Installed to $INSTALL_DIR/link-models"
echo "Run 'link-models --help' to get started"
'''
        
        # Create release directory
        release_dir = DIST_DIR / f"link-models-{get_version()}-linux"
        release_dir.mkdir(exist_ok=True)
        
        # Copy files
        shutil.copy(executable, release_dir / "link-models")
        
        install_script_path = release_dir / "install.sh"
        install_script_path.write_text(install_script)
        install_script_path.chmod(0o755)
        
        # Create tarball
        tarball = DIST_DIR / f"link-models-{get_version()}-linux.tar.gz"
        subprocess.run(
            ["tar", "-czf", str(tarball), "-C", str(DIST_DIR), release_dir.name],
            check=True,
        )
        
        print(f"Created: {tarball}")
    
    elif system == "Darwin":
        # Create macOS dmg or app bundle
        print("macOS installer creation not yet implemented")
    
    elif system == "Windows":
        # Create Windows installer (would need NSIS or WiX)
        print("Windows installer creation not yet implemented")


def main() -> None:
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Build link-models executable")
    parser.add_argument(
        "--backend",
        choices=["pyinstaller", "nuitka"],
        default="pyinstaller",
        help="Build backend to use",
    )
    parser.add_argument(
        "--onefile",
        action="store_true",
        default=True,
        help="Create single file executable (PyInstaller only)",
    )
    parser.add_argument(
        "--onedir",
        action="store_true",
        help="Create directory-based executable (PyInstaller only)",
    )
    parser.add_argument(
        "--clean",
        action="store_true",
        help="Clean build directories first",
    )
    parser.add_argument(
        "--installer",
        action="store_true",
        help="Create platform-specific installer",
    )
    
    args = parser.parse_args()
    
    version = get_version()
    print(f"Building link-models v{version}")
    print(f"Platform: {platform.system()} {platform.machine()}")
    
    if args.clean:
        clean_build_dirs()
    
    DIST_DIR.mkdir(parents=True, exist_ok=True)
    
    # Build
    if args.backend == "pyinstaller":
        onefile = not args.onedir
        executable = build_pyinstaller(onefile=onefile)
    else:
        executable = build_nuitka()
    
    # Create installer if requested
    if args.installer:
        create_installer(executable)
    
    print("\nBuild complete!")
    print(f"Executable: {executable}")
    print(f"Size: {executable.stat().st_size / 1024 / 1024:.2f} MB")


if __name__ == "__main__":
    main()
