"""Optimized GGUF metadata extraction with streaming parser.

This module provides both a streaming parser (for large files) and the
original numpy-based parser (for compatibility). The streaming parser
only reads the metadata header without loading tensor data, making it
much faster and memory-efficient for large GGUF files.
"""

from __future__ import annotations

import concurrent.futures
import os
import struct
from collections.abc import Callable
from pathlib import Path
from typing import Any

from .constants import GGUF_KEYS, SPECIAL_EOS_MARKERS
from .exceptions import GGUFError
from .logging import get_logger
from .models import GGUFMetadata

logger = get_logger(__name__)

# Maximum workers for parallel parsing
DEFAULT_MAX_WORKERS = min(8, (os.cpu_count() or 4) + 2)

# Timeout for parsing a single file (seconds)
PARSE_TIMEOUT = 60

# Maximum file size for numpy-based parser (1GB)
# Larger files use the streaming parser
MAX_NUMPY_FILE_SIZE = 1024 * 1024 * 1024

# GGUF format constants
GGUF_MAGIC = 0x46554747  # 'GGUF' in little-endian
GGUF_DEFAULT_ALIGNMENT = 32

# Value types
GGUF_TYPE_UINT8 = 0
GGUF_TYPE_INT8 = 1
GGUF_TYPE_UINT16 = 2
GGUF_TYPE_INT16 = 3
GGUF_TYPE_UINT32 = 4
GGUF_TYPE_INT32 = 5
GGUF_TYPE_FLOAT32 = 6
GGUF_TYPE_BOOL = 7
GGUF_TYPE_STRING = 8
GGUF_TYPE_ARRAY = 9
GGUF_TYPE_UINT64 = 10
GGUF_TYPE_INT64 = 11
GGUF_TYPE_FLOAT64 = 12


def _read_gguf_string(f: Any) -> str | None:
    """Read a GGUF string (8-byte length + data)."""
    len_bytes = f.read(8)
    if len(len_bytes) < 8:
        return None
    length = struct.unpack('<Q', len_bytes)[0]
    data = f.read(length)
    return data.decode('utf-8', errors='replace') if data else ""


def _read_value(f: Any, val_type: int) -> Any:
    """Read a value of given type from file."""
    if val_type == GGUF_TYPE_UINT8:
        return struct.unpack('<B', f.read(1))[0]
    elif val_type == GGUF_TYPE_INT8:
        return struct.unpack('<b', f.read(1))[0]
    elif val_type == GGUF_TYPE_UINT16:
        return struct.unpack('<H', f.read(2))[0]
    elif val_type == GGUF_TYPE_INT16:
        return struct.unpack('<h', f.read(2))[0]
    elif val_type == GGUF_TYPE_UINT32:
        return struct.unpack('<I', f.read(4))[0]
    elif val_type == GGUF_TYPE_INT32:
        return struct.unpack('<i', f.read(4))[0]
    elif val_type == GGUF_TYPE_FLOAT32:
        return struct.unpack('<f', f.read(4))[0]
    elif val_type == GGUF_TYPE_BOOL:
        return struct.unpack('<B', f.read(1))[0] != 0
    elif val_type == GGUF_TYPE_STRING:
        return _read_gguf_string(f)
    elif val_type == GGUF_TYPE_ARRAY:
        raw_itype = struct.unpack('<I', f.read(4))[0]
        alen = struct.unpack('<Q', f.read(8))[0]
        return [_read_value(f, raw_itype) for _ in range(alen)]
    elif val_type == GGUF_TYPE_UINT64:
        return struct.unpack('<Q', f.read(8))[0]
    elif val_type == GGUF_TYPE_INT64:
        return struct.unpack('<q', f.read(8))[0]
    elif val_type == GGUF_TYPE_FLOAT64:
        return struct.unpack('<d', f.read(8))[0]
    else:
        raise ValueError(f'Unknown GGUF value type: {val_type}')


def parse_gguf_metadata_streaming(path: Path) -> dict[str, Any]:
    """Parse only metadata from a GGUF file using streaming I/O.
    
    This is much more memory-efficient than GGUFReader because it:
    1. Only reads the header and metadata section
    2. Skips tensor info and tensor data entirely
    3. Uses simple file I/O instead of numpy/mmap
    
    Args:
        path: Path to GGUF file
        
    Returns:
        Dictionary of metadata key-value pairs
        
    Raises:
        GGUFError: If parsing fails
    """
    metadata: dict[str, Any] = {}

    try:
        with open(path, 'rb') as f:
            # Read header (24 bytes)
            # 4 bytes: magic
            # 4 bytes: version
            # 8 bytes: tensor_count (we skip tensors)
            # 8 bytes: metadata_kv_count
            header = f.read(4)
            if len(header) < 4:
                raise ValueError("File too small to be a valid GGUF file")

            magic = struct.unpack('<I', header)[0]
            if magic != GGUF_MAGIC:
                raise ValueError(f"Invalid GGUF magic: {magic:#x}")

            version_bytes = f.read(4)
            if len(version_bytes) < 4:
                raise ValueError("File truncated at version field")
            version = struct.unpack('<I', version_bytes)[0]

            if version not in (2, 3):
                raise ValueError(f"Unsupported GGUF version: {version}")

            # Read tensor_count (we don't use it, just skip)
            tensor_count_data = f.read(8)
            if len(tensor_count_data) < 8:
                raise ValueError("File truncated at tensor_count field")

            # Read metadata_kv_count
            kv_count_data = f.read(8)
            if len(kv_count_data) < 8:
                raise ValueError("File truncated at metadata_kv_count field")
            kv_count = struct.unpack('<Q', kv_count_data)[0]

            # Read metadata key-value pairs
            for _ in range(kv_count):
                # Read key
                key = _read_gguf_string(f)
                if key is None:
                    raise ValueError("File truncated while reading metadata key")

                # Read value type
                type_data = f.read(4)
                if len(type_data) < 4:
                    raise ValueError("File truncated while reading value type")
                val_type = struct.unpack('<I', type_data)[0]

                # Read value
                try:
                    value = _read_value(f, val_type)
                    metadata[key] = value
                except ValueError as e:
                    raise ValueError(f"Error reading value for key '{key}': {e}")

            return metadata

    except (OSError, struct.error) as e:
        raise GGUFError(
            f"Failed to parse GGUF file: {path}",
            details={"error": str(e), "path": str(path)},
        ) from e


def _extract_stop_tokens_from_metadata(metadata: dict[str, Any]) -> list[str]:
    """Extract stop tokens from metadata dictionary.
    
    Args:
        metadata: Dictionary of GGUF metadata
        
    Returns:
        List of stop token strings
    """
    stop_tokens: list[str] = []

    # Try to get EOS token ID and look up token
    eos_token_id_key = GGUF_KEYS["eos_token_id"]
    assert isinstance(eos_token_id_key, str)  # for type checker
    eos_token_id = metadata.get(eos_token_id_key)

    tokens_key = GGUF_KEYS["tokens"]
    assert isinstance(tokens_key, str)  # for type checker
    tokens = metadata.get(tokens_key)

    if eos_token_id is not None and tokens and isinstance(tokens, list):
        if isinstance(eos_token_id, int) and eos_token_id < len(tokens):
            token = tokens[eos_token_id]
            if isinstance(token, str) and token and token not in stop_tokens:
                stop_tokens.append(token)

    # Scan for special EOS markers in tokens
    if tokens and isinstance(tokens, list):
        for token in tokens:
            if isinstance(token, str):
                for marker in SPECIAL_EOS_MARKERS:
                    if marker in token and token not in stop_tokens:
                        stop_tokens.append(token)
                        break

    return stop_tokens


def parse_gguf_file(path: Path, use_streaming: bool = True) -> GGUFMetadata:
    """Parse metadata from a single GGUF file.
    
    Args:
        path: Path to GGUF file
        use_streaming: Use streaming parser (faster, less memory) instead of
                      numpy-based parser. Defaults to True.
        
    Returns:
        Extracted metadata
        
    Raises:
        GGUFError: If parsing fails
    """
    try:
        # Always use streaming parser - it's faster and uses less memory
        if use_streaming:
            metadata = parse_gguf_metadata_streaming(path)

            # Extract architecture first (needed for other lookups)
            arch_key = GGUF_KEYS["architecture"]
            assert isinstance(arch_key, str)
            architecture = metadata.get(arch_key)

            # Context length - try architecture-specific first
            context_length = None
            if architecture:
                arch_ctx_key = f"{architecture}.context_length"
                context_length = metadata.get(arch_ctx_key)

            if context_length is None:
                ctx_keys = GGUF_KEYS["context_length"]
                assert isinstance(ctx_keys, list)
                for key in ctx_keys:
                    if "{arch}" in key and architecture:
                        key = key.format(arch=architecture)
                    assert isinstance(key, str)
                    context_length = metadata.get(key)
                    if context_length is not None:
                        break

            # Block count / layers
            if architecture:
                block_count_key = f"{architecture}.block_count"
            else:
                block_count_key = "llama.block_count"
            num_layers = metadata.get(block_count_key)

            # Get file_type (quantization)
            file_type_key = GGUF_KEYS["file_type"]
            assert isinstance(file_type_key, str)
            file_type = metadata.get(file_type_key)

            # Get other simple keys
            name_key = GGUF_KEYS["name"]
            assert isinstance(name_key, str)

            vocab_key = GGUF_KEYS["vocab_size"]
            assert isinstance(vocab_key, str)

            chat_key = GGUF_KEYS["chat_template"]
            assert isinstance(chat_key, str)

            return GGUFMetadata(
                architecture=architecture,
                name=metadata.get(name_key),
                context_length=context_length,
                quantization=file_type,
                vocab_size=metadata.get(vocab_key),
                num_hidden_layers=num_layers,
                chat_template=metadata.get(chat_key),
                stop_tokens=_extract_stop_tokens_from_metadata(metadata),
            )
        else:
            # Fallback to numpy-based parser (for compatibility/testing)
            return _parse_gguf_numpy(path)

    except GGUFError:
        raise
    except Exception as e:
        raise GGUFError(
            f"Failed to parse GGUF file: {path}",
            details={"error": str(e), "path": str(path)},
        ) from e


def _parse_gguf_numpy(path: Path) -> GGUFMetadata:
    """Parse GGUF file using numpy/gguf library (legacy method).
    
    This is kept for compatibility but is slower and uses more memory.
    """
    import gguf

    reader = gguf.GGUFReader(str(path), "r")

    def _get_field(reader: gguf.GGUFReader, key: str, is_str: bool = False) -> Any:
        """Safely extract a field from GGUF reader."""
        try:
            field = reader.get_field(key)
            if field is None:
                return None
            if not hasattr(field, 'data') or not field.data:
                return None

            idx = field.data[0]
            if idx is None or idx >= len(field.parts):
                return None

            val = field.parts[idx]

            if is_str:
                if isinstance(val, (bytes, bytearray)):
                    return val.decode('utf-8', errors='replace')
                return str(val)
            return val
        except (IndexError, AttributeError, UnicodeDecodeError):
            return None

    # Extract architecture first (needed for other lookups)
    arch_key = GGUF_KEYS["architecture"]
    assert isinstance(arch_key, str)
    architecture = _get_field(reader, arch_key, is_str=True)

    # Context length - try architecture-specific first
    context_length = None
    if architecture:
        arch_ctx_key = f"{architecture}.context_length"
        context_length = _get_field(reader, arch_ctx_key)

    if context_length is None:
        ctx_keys = GGUF_KEYS["context_length"]
        assert isinstance(ctx_keys, list)
        for key in ctx_keys:
            if "{arch}" in key and architecture:
                key = key.format(arch=architecture)
            assert isinstance(key, str)
            context_length = _get_field(reader, key)
            if context_length is not None:
                break

    # Block count / layers
    if architecture:
        block_count_key = f"{architecture}.block_count"
    else:
        block_count_key = "llama.block_count"
    num_layers = _get_field(reader, block_count_key)

    # Simple string keys
    name_key = GGUF_KEYS["name"]
    assert isinstance(name_key, str)

    file_type_key = GGUF_KEYS["file_type"]
    assert isinstance(file_type_key, str)

    vocab_key = GGUF_KEYS["vocab_size"]
    assert isinstance(vocab_key, str)

    chat_key = GGUF_KEYS["chat_template"]
    assert isinstance(chat_key, str)

    return GGUFMetadata(
        architecture=architecture,
        name=_get_field(reader, name_key, is_str=True),
        context_length=context_length,
        quantization=_get_field(reader, file_type_key),
        vocab_size=_get_field(reader, vocab_key),
        num_hidden_layers=num_layers,
        chat_template=_get_field(reader, chat_key, is_str=True),
        stop_tokens=_extract_stop_tokens_numpy(reader),
    )


def _extract_stop_tokens_numpy(reader: Any) -> list[str]:
    """Extract stop tokens from GGUF reader (numpy-based)."""
    stop_tokens: list[str] = []

    # Try to get EOS token ID and look up token
    eos_field = reader.get_field(GGUF_KEYS["eos_token_id"])
    if eos_field and hasattr(eos_field, 'data') and eos_field.data:
        tokens_field = reader.get_field(GGUF_KEYS["tokens"])
        if tokens_field and hasattr(tokens_field, 'parts'):
            for idx in eos_field.data:
                try:
                    if isinstance(idx, (list, tuple)):
                        for i in idx:
                            if i < len(tokens_field.data):
                                token_idx = tokens_field.data[i]
                                token_bytes = tokens_field.parts[token_idx]
                                if isinstance(token_bytes, bytes):
                                    token_str = token_bytes.decode('utf-8', errors='replace')
                                else:
                                    token_str = str(token_bytes)
                                if token_str and token_str not in stop_tokens:
                                    stop_tokens.append(token_str)
                    elif idx < len(tokens_field.data):
                        token_idx = tokens_field.data[idx]
                        token_bytes = tokens_field.parts[token_idx]
                        if isinstance(token_bytes, bytes):
                            token_str = token_bytes.decode('utf-8', errors='replace')
                        else:
                            token_str = str(token_bytes)
                        if token_str and token_str not in stop_tokens:
                            stop_tokens.append(token_str)
                except (IndexError, UnicodeDecodeError):
                    continue

    # Scan for special EOS markers
    tokens_field = reader.get_field(GGUF_KEYS["tokens"])
    if tokens_field and hasattr(tokens_field, 'parts'):
        try:
            for i, idx in enumerate(tokens_field.data):
                if idx >= len(tokens_field.parts):
                    continue
                raw_bytes = tokens_field.parts[idx]
                if isinstance(raw_bytes, bytes):
                    token_str = raw_bytes.decode('utf-8', errors='replace')
                else:
                    token_str = str(raw_bytes)

                for marker in SPECIAL_EOS_MARKERS:
                    if marker in token_str and token_str not in stop_tokens:
                        stop_tokens.append(token_str)
                        break
        except (IndexError, UnicodeDecodeError):
            pass

    return stop_tokens


class ParallelGGUFParser:
    """Parallel GGUF metadata extraction for multiple files."""

    def __init__(self, max_workers: int | None = None) -> None:
        """Initialize parser.
        
        Args:
            max_workers: Maximum parallel workers (defaults to CPU count + 2)
        """
        self.max_workers = max_workers or DEFAULT_MAX_WORKERS
        self._executor: concurrent.futures.ProcessPoolExecutor | None = None

    def __enter__(self) -> ParallelGGUFParser:
        """Context manager entry."""
        self._executor = concurrent.futures.ProcessPoolExecutor(
            max_workers=self.max_workers,
        )
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        """Context manager exit."""
        if self._executor:
            self._executor.shutdown(wait=True)
            self._executor = None

    def parse_files(
        self,
        paths: list[Path],
        *,
        progress_callback: Callable[[int, int], None] | None = None,
        use_threads: bool = False,
    ) -> dict[Path, GGUFMetadata | None]:
        """Parse multiple GGUF files in parallel.
        
        Args:
            paths: List of paths to parse
            progress_callback: Optional callback(progress, total) for updates
            use_threads: Use ThreadPoolExecutor instead of ProcessPoolExecutor
                        (better for I/O bound, worse for CPU bound)
            
        Returns:
            Dictionary mapping paths to metadata (None if parsing failed)
        """
        if not paths:
            return {}

        results: dict[Path, GGUFMetadata | None] = {}
        total = len(paths)

        # Use sequential processing for small numbers of files
        # The streaming parser is already very fast, so parallel overhead
        # may not be worth it for small batches
        if len(paths) <= 4:
            for path in paths:
                try:
                    results[path] = parse_gguf_file(path)
                except GGUFError as e:
                    logger.warning("Failed to parse file", path=str(path), error=str(e))
                    results[path] = None

                if progress_callback:
                    progress_callback(len(results), total)
            return results

        # Use parallel processing for larger batches
        if use_threads:
            # ThreadPoolExecutor is better for I/O bound workloads
            return self._parse_with_threads(paths, results, progress_callback, total)
        else:
            # ProcessPoolExecutor is better for CPU bound workloads
            return self._parse_with_processes(paths, results, progress_callback, total)

    def _parse_with_threads(
        self,
        paths: list[Path],
        results: dict[Path, GGUFMetadata | None],
        progress_callback: Callable[[int, int], None] | None,
        total: int,
    ) -> dict[Path, GGUFMetadata | None]:
        """Parse files using ThreadPoolExecutor (I/O bound)."""
        with concurrent.futures.ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            futures = {executor.submit(parse_gguf_file, path): path for path in paths}

            for future in concurrent.futures.as_completed(futures):
                path = futures[future]
                try:
                    results[path] = future.result(timeout=PARSE_TIMEOUT)
                except concurrent.futures.TimeoutError:
                    logger.warning("Parsing timed out", path=str(path))
                    results[path] = None
                except Exception as e:
                    logger.warning("Failed to parse file", path=str(path), error=str(e))
                    results[path] = None

                if progress_callback:
                    progress_callback(len(results), total)

        return results

    def _parse_with_processes(
        self,
        paths: list[Path],
        results: dict[Path, GGUFMetadata | None],
        progress_callback: Callable[[int, int], None] | None,
        total: int,
    ) -> dict[Path, GGUFMetadata | None]:
        """Parse files using ProcessPoolExecutor (CPU bound)."""
        if not self._executor:
            raise RuntimeError("Parser not started. Use as context manager.")

        futures = {
            self._executor.submit(_parse_worker, path): path
            for path in paths
        }

        for future in concurrent.futures.as_completed(futures):
            path = futures[future]
            try:
                results[path] = future.result(timeout=PARSE_TIMEOUT)
            except concurrent.futures.TimeoutError:
                logger.warning("Parsing timed out", path=str(path))
                results[path] = None
            except Exception as e:
                logger.warning("Failed to parse file", path=str(path), error=str(e))
                results[path] = None

            if progress_callback:
                progress_callback(len(results), total)

        return results

    def parse_single(self, path: Path) -> GGUFMetadata | None:
        """Parse a single file.
        
        Args:
            path: Path to GGUF file
            
        Returns:
            Metadata or None if parsing failed
        """
        try:
            return parse_gguf_file(path)
        except GGUFError as e:
            logger.warning("Failed to parse file", path=str(path), error=str(e))
            return None


def _parse_worker(path: Path) -> GGUFMetadata | None:
    """Worker function for process pool (must be top-level for pickle)."""
    try:
        return parse_gguf_file(path)
    except Exception:
        return None


# Convenience function for simple use cases
def parse_gguf_files(
    paths: list[Path],
    *,
    max_workers: int | None = None,
) -> dict[Path, GGUFMetadata | None]:
    """Parse multiple GGUF files in parallel.
    
    Args:
        paths: List of paths to parse
        max_workers: Maximum parallel workers
        
    Returns:
        Dictionary mapping paths to metadata
    """
    with ParallelGGUFParser(max_workers=max_workers) as parser:
        return parser.parse_files(paths)
