"""Unit tests for core models."""

from __future__ import annotations

from pathlib import Path

import pytest

from link_models.core.models import (
    GGUFMetadata,
    ModelInfo,
    ModelGroup,
    normalize_model_id,
    get_multipart_base,
    is_partial_download,
    get_real_filename,
    strip_quantization_suffix,
    get_mmproj_base,
)


class TestNormalizeModelId:
    """Tests for normalize_model_id function."""
    
    def test_lowercase_conversion(self) -> None:
        assert normalize_model_id("MyModel") == "mymodel"
    
    def test_special_chars_replaced(self) -> None:
        assert normalize_model_id("my_model@123") == "my-model-123"
    
    def test_multiple_special_chars_collapsed(self) -> None:
        assert normalize_model_id("my___model") == "my-model"
    
    def test_leading_trailing_hyphens_removed(self) -> None:
        assert normalize_model_id("-my-model-") == "my-model"
    
    def test_empty_result_returns_unknown(self) -> None:
        assert normalize_model_id("!@#$") == "unknown"
    
    def test_real_model_names(self) -> None:
        assert normalize_model_id("Llama-2-7B-chat-GGUF") == "llama-2-7b-chat-gguf"
        assert normalize_model_id("Mixtral-8x7B-Instruct-v0.1") == "mixtral-8x7b-instruct-v0-1"


class TestGetMultipartBase:
    """Tests for get_multipart_base function."""
    
    def test_detects_multipart(self) -> None:
        assert get_multipart_base("model-00001-of-00003.gguf") == "model"
    
    def test_detects_multipart_variation(self) -> None:
        assert get_multipart_base("model-001-of-003.gguf") == "model"
    
    def test_single_file_returns_none(self) -> None:
        assert get_multipart_base("model.gguf") is None
    
    def test_similar_names_not_detected(self) -> None:
        assert get_multipart_base("model-123.gguf") is None
    
    def test_case_insensitive(self) -> None:
        assert get_multipart_base("MODEL-00001-OF-00003.GGUF") == "MODEL"


class TestIsPartialDownload:
    """Tests for is_partial_download function."""
    
    def test_part_extension(self) -> None:
        assert is_partial_download("model.gguf.part") is True
    
    def test_tmp_extension(self) -> None:
        assert is_partial_download("model.gguf.tmp") is True
    
    def test_crdownload_extension(self) -> None:
        assert is_partial_download("model.gguf.crdownload") is True
    
    def test_complete_file_not_partial(self) -> None:
        assert is_partial_download("model.gguf") is False
    
    def test_normal_file_with_tmp_in_name(self) -> None:
        assert is_partial_download("my_tmp_model.gguf") is False


class TestGetRealFilename:
    """Tests for get_real_filename function."""
    
    def test_removes_part_extension(self) -> None:
        assert get_real_filename("model.gguf.part") == "model.gguf"
    
    def test_removes_tmp_extension(self) -> None:
        assert get_real_filename("model.gguf.tmp") == "model.gguf"
    
    def test_no_change_for_complete(self) -> None:
        assert get_real_filename("model.gguf") == "model.gguf"
    
    def test_multiple_extensions(self) -> None:
        assert get_real_filename("model.gguf.part.tmp") == "model.gguf.part"


class TestStripQuantizationSuffix:
    """Tests for strip_quantization_suffix function."""
    
    def test_strip_q4_k_m(self) -> None:
        assert strip_quantization_suffix("model-Q4_K_M.gguf") == "model"
    
    def test_strip_q5_k_s(self) -> None:
        assert strip_quantization_suffix("model-Q5_K_S.gguf") == "model"
    
    def test_strip_f16(self) -> None:
        assert strip_quantization_suffix("model-F16.gguf") == "model"
    
    def test_strip_bf16(self) -> None:
        assert strip_quantization_suffix("model-BF16.gguf") == "model"
    
    def test_no_quantization_unchanged(self) -> None:
        assert strip_quantization_suffix("model.gguf") == "model.gguf"
    
    def test_iq_quantization(self) -> None:
        assert strip_quantization_suffix("model-IQ4_XS.gguf") == "model"


class TestGetMmprojBase:
    """Tests for get_mmproj_base function."""
    
    def test_mmproj_prefix(self) -> None:
        assert get_mmproj_base("mmproj-model.gguf") == "model"
    
    def test_mmproj_prefix_with_quant(self) -> None:
        assert get_mmproj_base("mmproj-model-f16.gguf") == "model"
    
    def test_mmproj_suffix(self) -> None:
        assert get_mmproj_base("model-mmproj.gguf") == "model"
    
    def test_no_mmproj_returns_none(self) -> None:
        assert get_mmproj_base("model.gguf") is None


class TestGGUFMetadata:
    """Tests for GGUFMetadata dataclass."""
    
    def test_default_backend_unknown(self) -> None:
        metadata = GGUFMetadata()
        assert metadata.get_backend() == "llama-cpp"
    
    def test_backend_from_architecture(self) -> None:
        metadata = GGUFMetadata(architecture="whisper")
        assert metadata.get_backend() == "whisper"
    
    def test_backend_llama_variant(self) -> None:
        metadata = GGUFMetadata(architecture="llama2")
        assert metadata.get_backend() == "llama-cpp"


class TestModelInfo:
    """Tests for ModelInfo dataclass."""
    
    def test_is_gguf_true(self, temp_dir: Path) -> None:
        path = temp_dir / "model.gguf"
        info = ModelInfo(path=path)
        assert info.is_gguf is True
    
    def test_is_gguf_false(self, temp_dir: Path) -> None:
        path = temp_dir / "model.bin"
        info = ModelInfo(path=path)
        assert info.is_gguf is False
    
    def test_is_mmproj_true(self, temp_dir: Path) -> None:
        path = temp_dir / "mmproj-model.gguf"
        info = ModelInfo(path=path)
        assert info.is_mmproj is True
    
    def test_is_mmproj_false(self, temp_dir: Path) -> None:
        path = temp_dir / "model.gguf"
        info = ModelInfo(path=path)
        assert info.is_mmproj is False


class TestModelGroup:
    """Tests for ModelGroup dataclass."""
    
    def test_is_multipart_true(self, temp_dir: Path) -> None:
        files = [
            ModelInfo(path=temp_dir / "model-00001-of-00002.gguf"),
            ModelInfo(path=temp_dir / "model-00002-of-00002.gguf"),
        ]
        group = ModelGroup(base_name="model", files=files)
        assert group.is_multipart is True
    
    def test_is_multipart_false(self, temp_dir: Path) -> None:
        files = [ModelInfo(path=temp_dir / "model.gguf")]
        group = ModelGroup(base_name="model", files=files)
        assert group.is_multipart is False
    
    def test_primary_file_single(self, temp_dir: Path) -> None:
        file_info = ModelInfo(path=temp_dir / "model.gguf")
        group = ModelGroup(base_name="model", files=[file_info])
        assert group.primary_file == file_info
    
    def test_primary_file_multipart(self, temp_dir: Path) -> None:
        files = [
            ModelInfo(path=temp_dir / "model-00002-of-00002.gguf"),
            ModelInfo(path=temp_dir / "model-00001-of-00002.gguf"),
        ]
        group = ModelGroup(base_name="model", files=files)
        # Should return first part when sorted
        assert "00001" in group.primary_file.name
    
    def test_has_vision_true(self, temp_dir: Path) -> None:
        files = [ModelInfo(path=temp_dir / "model.gguf")]
        mmproj = ModelInfo(path=temp_dir / "mmproj-model.gguf")
        group = ModelGroup(base_name="model", files=files, mmproj_file=mmproj)
        assert group.has_vision is True
    
    def test_has_vision_false(self, temp_dir: Path) -> None:
        files = [ModelInfo(path=temp_dir / "model.gguf")]
        group = ModelGroup(base_name="model", files=files)
        assert group.has_vision is False
    
    def test_get_all_files(self, temp_dir: Path) -> None:
        files = [ModelInfo(path=temp_dir / "model.gguf")]
        mmproj = ModelInfo(path=temp_dir / "mmproj-model.gguf")
        group = ModelGroup(base_name="model", files=files, mmproj_file=mmproj)
        all_files = group.get_all_files()
        assert len(all_files) == 2
