"""Pytest configuration and fixtures."""

from __future__ import annotations

import tempfile
from pathlib import Path
from typing import Generator

import pytest


@pytest.fixture
def temp_dir() -> Generator[Path, None, None]:
    """Create a temporary directory for tests."""
    with tempfile.TemporaryDirectory() as tmp:
        yield Path(tmp)


@pytest.fixture
def mock_gguf_file(temp_dir: Path) -> Path:
    """Create a mock GGUF file for testing.
    
    Note: This creates a minimal file that looks like a GGUF
    but won't pass actual GGUF parsing. Use for filesystem tests only.
    """
    gguf_path = temp_dir / "test-model-q4_k_m.gguf"
    
    # Write minimal GGUF header magic
    # GGUF magic number: 0x46554747 "GGUF"
    with open(gguf_path, "wb") as f:
        f.write(b"GGUF")  # Magic
        f.write(b"\x03\x00\x00\x00")  # Version 3
        f.write(b"\x00\x00\x00\x00\x00\x00\x00\x00")  # Tensor count (0)
        f.write(b"\x01\x00\x00\x00\x00\x00\x00\x00")  # Metadata count (1)
        # Add a minimal metadata kv pair
        f.write(b"\x03\x00\x00\x00")  # Key length (3)
        f.write(b"foo")  # Key
        f.write(b"\x08\x00")  # Value type (string)
        f.write(b"\x03\x00\x00\x00")  # String length
        f.write(b"bar")  # String value
    
    return gguf_path


@pytest.fixture
def source_dir(temp_dir: Path) -> Path:
    """Create a source directory with some test models."""
    src = temp_dir / "models"
    src.mkdir()
    
    # Create some test files
    (src / "model1-q4_k_m.gguf").write_bytes(b"GGUF" + b"\x00" * 100)
    (src / "model2-q5_k_m.gguf").write_bytes(b"GGUF" + b"\x00" * 100)
    (src / "model3-00001-of-00002.gguf").write_bytes(b"GGUF" + b"\x00" * 100)
    (src / "model3-00002-of-00002.gguf").write_bytes(b"GGUF" + b"\x00" * 100)
    
    return src


@pytest.fixture
def config_dict(temp_dir: Path) -> dict:
    """Create a default configuration dictionary."""
    return {
        "source_dir": str(temp_dir / "models"),
        "backends": {
            "llama_cpp": {
                "enabled": True,
                "output_dir": str(temp_dir / "llama_models"),
            },
            "localai": {
                "enabled": True,
                "output_dir": str(temp_dir / "localai_models"),
            },
        },
        "watch": {
            "enabled": False,
            "check_interval": 0.1,
        },
        "logging": {
            "level": "DEBUG",
        },
    }
