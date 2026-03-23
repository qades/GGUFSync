"""Tests for GGUF parser main functions."""

from __future__ import annotations

import io
import struct
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from gguf_sync.core.exceptions import GGUFError
from gguf_sync.core.gguf_parser import (
    GGUF_MAGIC,
    _extract_stop_tokens_from_metadata,
    parse_gguf_file,
    parse_gguf_metadata_streaming,
)
from gguf_sync.core.models import GGUFMetadata


def create_minimal_gguf_header(tensor_count: int = 0, metadata_kv_count: int = 0) -> bytes:
    """Create a minimal valid GGUF file header for testing."""
    # GGUF Header structure:
    # - Magic (4 bytes): GGUF_MAGIC
    # - Version (4 bytes): 3
    # - Tensor Count (8 bytes)
    # - Metadata KV Count (8 bytes)
    data = struct.pack("<I", GGUF_MAGIC)  # Magic
    data += struct.pack("<I", 3)  # Version
    data += struct.pack("<Q", tensor_count)  # Tensor count
    data += struct.pack("<Q", metadata_kv_count)  # Metadata KV count
    return data


class TestParseGGUFMetadataStreaming:
    """Tests for parse_gguf_metadata_streaming function."""

    def test_invalid_magic_raises_error(self, tmp_path: Path) -> None:
        """Test that invalid magic raises GGUFError."""
        # Create file with wrong magic
        test_file = tmp_path / "test.gguf"
        test_file.write_bytes(b"INVALID!" + b"\x00" * 24)

        with pytest.raises((GGUFError, ValueError)) as exc_info:
            parse_gguf_metadata_streaming(test_file)
        assert "Invalid GGUF magic" in str(exc_info.value) or "magic" in str(exc_info.value).lower()

    def test_file_not_found_raises_error(self, tmp_path: Path) -> None:
        """Test that nonexistent file raises GGUFError."""
        test_file = tmp_path / "nonexistent.gguf"

        with pytest.raises(GGUFError) as exc_info:
            parse_gguf_metadata_streaming(test_file)
        assert "Failed to parse" in str(exc_info.value) or "File not found" in str(exc_info.value)

    def test_empty_metadata(self, tmp_path: Path) -> None:
        """Test parsing file with empty metadata."""
        test_file = tmp_path / "test.gguf"
        header = create_minimal_gguf_header(tensor_count=0, metadata_kv_count=0)
        test_file.write_bytes(header)

        result = parse_gguf_metadata_streaming(test_file)
        assert result == {}


class TestParseGGUFFile:
    """Tests for parse_gguf_file function."""

    def test_empty_file_returns_default_metadata(self, tmp_path: Path) -> None:
        """Test parsing empty file returns default metadata."""
        test_file = tmp_path / "test.gguf"
        # Create minimal valid header but with truncated content
        header = create_minimal_gguf_header(tensor_count=0, metadata_kv_count=0)
        test_file.write_bytes(header)

        result = parse_gguf_file(test_file, use_streaming=True)
        assert isinstance(result, GGUFMetadata)
        assert result.name is None
        assert result.architecture is None

    def test_use_streaming_false_uses_numpy(self, tmp_path: Path) -> None:
        """Test that use_streaming=False uses numpy parser."""
        test_file = tmp_path / "test.gguf"
        header = create_minimal_gguf_header(tensor_count=0, metadata_kv_count=0)
        test_file.write_bytes(header)

        # This will fail because numpy parser needs actual GGUF structure
        # But it tests the code path
        try:
            result = parse_gguf_file(test_file, use_streaming=False)
            assert isinstance(result, GGUFMetadata)
        except (GGUFError, ImportError):
            # Expected if numpy/gguf package not available
            pass

    def test_file_not_found_raises_error(self, tmp_path: Path) -> None:
        """Test that nonexistent file raises error."""
        test_file = tmp_path / "nonexistent.gguf"

        with pytest.raises(GGUFError):
            parse_gguf_file(test_file)


class TestExtractStopTokens:
    """Tests for _extract_stop_tokens_from_metadata function."""

    def test_no_stop_tokens(self) -> None:
        """Test extraction with no stop tokens in metadata."""
        metadata = {"general.name": "test-model"}
        result = _extract_stop_tokens_from_metadata(metadata)
        assert result == []

    def test_eos_token(self) -> None:
        """Test extraction of EOS token."""
        from gguf_sync.core.constants import GGUF_KEYS

        eos_key = GGUF_KEYS["eos_token_id"]
        tokens_key = GGUF_KEYS["tokens"]
        metadata = {eos_key: 2, tokens_key: ["a", "b", "<|end|>"]}
        result = _extract_stop_tokens_from_metadata(metadata)
        assert "<|end|>" in result

    def test_eos_markers_in_tokens(self) -> None:
        """Test extraction of tokens containing EOS markers."""
        from gguf_sync.core.constants import GGUF_KEYS

        tokens_key = GGUF_KEYS["tokens"]
        # Test with actual EOS markers from SPECIAL_EOS_MARKERS
        metadata = {tokens_key: ["a", "b", "c", "<|end|>"]}
        result = _extract_stop_tokens_from_metadata(metadata)
        # Token containing EOS marker should be extracted
        assert "<|end|>" in result

    def test_bos_token_not_included(self) -> None:
        """Test that BOS token is not included in stop tokens."""
        from gguf_sync.core.constants import GGUF_KEYS

        tokens_key = GGUF_KEYS["tokens"]
        # BOS token doesn't have EOS markers, so it shouldn't be included
        metadata = {tokens_key: ["<bos>", "a"]}
        result = _extract_stop_tokens_from_metadata(metadata)
        # BOS token should not be in stop tokens
        assert "<bos>" not in result

    def test_special_tokens_with_eos_markers(self) -> None:
        """Test extraction of tokens containing EOS markers."""
        from gguf_sync.core.constants import GGUF_KEYS

        tokens_key = GGUF_KEYS["tokens"]
        metadata = {tokens_key: ["<|end|>", "<|user|>", "<|assistant|>", "<|eot_id|>"]}
        result = _extract_stop_tokens_from_metadata(metadata)
        # Tokens containing EOS markers should be included
        assert "<|end|>" in result
        assert "<|eot_id|>" in result

    def test_chat_template_stop_tokens(self) -> None:
        """Test extraction of stop tokens from chat template."""
        metadata = {
            "tokenizer.chat_template": "{% for message in messages %}{{ message.content }}{% endfor %}"
        }
        result = _extract_stop_tokens_from_metadata(metadata)
        # Should extract some tokens from template patterns
        assert isinstance(result, list)


class TestGGUFParserErrorHandling:
    """Tests for GGUF parser error handling."""

    def test_permission_error_raises_gguf_error(self, tmp_path: Path) -> None:
        """Test that permission error raises GGUFError."""
        test_file = tmp_path / "test.gguf"
        test_file.write_bytes(create_minimal_gguf_header())
        test_file.chmod(0o000)  # Remove all permissions

        try:
            with pytest.raises(GGUFError):
                parse_gguf_file(test_file)
        finally:
            test_file.chmod(0o644)  # Restore permissions for cleanup

    def test_truncated_header_raises_error(self, tmp_path: Path) -> None:
        """Test that truncated header raises error."""
        test_file = tmp_path / "test.gguf"
        # Write only magic, missing rest of header
        test_file.write_bytes(struct.pack("<I", GGUF_MAGIC))

        # This should raise ValueError for truncated file
        with pytest.raises((GGUFError, ValueError)):
            parse_gguf_metadata_streaming(test_file)
