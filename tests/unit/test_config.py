"""Unit tests for configuration loading."""

from __future__ import annotations

import os
from pathlib import Path

import pytest
import yaml

from link_models.core.config import (
    ConfigLoader,
    load_yaml_config,
    get_env_config,
    merge_configs,
    _parse_env_value,
)
from link_models.core.exceptions import ConfigError
from link_models.core.models import AppConfig, LlamaCppConfig, LocalAIConfig


class TestLoadYamlConfig:
    """Tests for load_yaml_config function."""
    
    def test_load_valid_yaml(self, temp_dir: Path) -> None:
        config_path = temp_dir / "config.yaml"
        config_path.write_text("key: value\nlist:\n  - item1\n  - item2")
        
        result = load_yaml_config(config_path)
        assert result == {"key": "value", "list": ["item1", "item2"]}
    
    def test_load_empty_yaml(self, temp_dir: Path) -> None:
        config_path = temp_dir / "config.yaml"
        config_path.write_text("")
        
        result = load_yaml_config(config_path)
        assert result == {}
    
    def test_file_not_found(self, temp_dir: Path) -> None:
        with pytest.raises(ConfigError) as exc_info:
            load_yaml_config(temp_dir / "nonexistent.yaml")
        assert "not found" in str(exc_info.value.message)
    
    def test_invalid_yaml(self, temp_dir: Path) -> None:
        config_path = temp_dir / "config.yaml"
        config_path.write_text("invalid: yaml: content:")
        
        with pytest.raises(ConfigError) as exc_info:
            load_yaml_config(config_path)
        assert "Invalid YAML" in str(exc_info.value.message)


class TestParseEnvValue:
    """Tests for _parse_env_value function."""
    
    def test_boolean_true_values(self) -> None:
        assert _parse_env_value("true") is True
        assert _parse_env_value("True") is True
        assert _parse_env_value("yes") is True
        assert _parse_env_value("1") is True
        assert _parse_env_value("on") is True
    
    def test_boolean_false_values(self) -> None:
        assert _parse_env_value("false") is False
        assert _parse_env_value("False") is False
        assert _parse_env_value("no") is False
        assert _parse_env_value("0") is False
        assert _parse_env_value("off") is False
    
    def test_integer_parsing(self) -> None:
        assert _parse_env_value("42") == 42
        assert _parse_env_value("-10") == -10
    
    def test_float_parsing(self) -> None:
        assert _parse_env_value("3.14") == 3.14
        assert _parse_env_value("-0.5") == -0.5
    
    def test_path_expansion(self) -> None:
        assert _parse_env_value("~/models") == os.path.expanduser("~/models")
    
    def test_string_default(self) -> None:
        assert _parse_env_value("hello") == "hello"


class TestGetEnvConfig:
    """Tests for get_env_config function."""
    
    def test_single_level_var(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("LINK_MODELS_SOURCE_DIR", "/models")
        
        result = get_env_config()
        assert result == {"source_dir": "/models"}
    
    def test_nested_var(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("LINK_MODELS_BACKENDS__LLAMA_CPP__ENABLED", "true")
        
        result = get_env_config()
        assert result == {"backends": {"llama_cpp": {"enabled": True}}}
    
    def test_multiple_vars(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("LINK_MODELS_SOURCE_DIR", "/models")
        monkeypatch.setenv("LINK_MODELS_LOGGING__LEVEL", "DEBUG")
        
        result = get_env_config()
        assert result["source_dir"] == "/models"
        assert result["logging"]["level"] == "DEBUG"
    
    def test_ignores_non_prefixed(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("OTHER_VAR", "value")
        monkeypatch.setenv("LINK_MODELS_KEY", "value2")
        
        result = get_env_config()
        assert "other_var" not in result
        assert result["key"] == "value2"


class TestMergeConfigs:
    """Tests for merge_configs function."""
    
    def test_simple_merge(self) -> None:
        base = {"a": 1, "b": 2}
        override = {"b": 3, "c": 4}
        
        result = merge_configs(base, override)
        assert result == {"a": 1, "b": 3, "c": 4}
    
    def test_deep_merge(self) -> None:
        base = {"nested": {"a": 1, "b": 2}}
        override = {"nested": {"b": 3, "c": 4}}
        
        result = merge_configs(base, override)
        assert result == {"nested": {"a": 1, "b": 3, "c": 4}}
    
    def test_override_non_dict(self) -> None:
        base = {"key": {"nested": "value"}}
        override = {"key": "replaced"}
        
        result = merge_configs(base, override)
        assert result == {"key": "replaced"}


class TestConfigLoader:
    """Tests for ConfigLoader class."""
    
    def test_load_from_file(self, temp_dir: Path) -> None:
        config_path = temp_dir / "config.yaml"
        config_data = {
            "source_dir": "/custom/models",
            "backends": {
                "llama_cpp": {
                    "enabled": True,
                    "output_dir": "/custom/llama",
                }
            }
        }
        config_path.write_text(yaml.dump(config_data))
        
        loader = ConfigLoader()
        config = loader.load(config_path=config_path)
        
        assert isinstance(config, AppConfig)
        assert str(config.source_dir) == "/custom/models"
        assert "llama_cpp" in config.backends
    
    def test_cli_args_override_file(self, temp_dir: Path) -> None:
        config_path = temp_dir / "config.yaml"
        config_data = {"source_dir": "/file/models"}
        config_path.write_text(yaml.dump(config_data))
        
        loader = ConfigLoader()
        config = loader.load(
            config_path=config_path,
            cli_args={"source_dir": "/cli/models"}
        )
        
        assert str(config.source_dir) == "/cli/models"
    
    def test_env_vars_override_file(
        self,
        temp_dir: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        config_path = temp_dir / "config.yaml"
        config_data = {"source_dir": "/file/models"}
        config_path.write_text(yaml.dump(config_data))
        
        monkeypatch.setenv("LINK_MODELS_SOURCE_DIR", "/env/models")
        
        loader = ConfigLoader()
        config = loader.load(config_path=config_path)
        
        # CLI args take precedence over env vars
        # But since we didn't provide CLI args, env should work
        assert str(config.source_dir) == "/env/models"
    
    def test_default_backends_when_none_specified(self, temp_dir: Path) -> None:
        loader = ConfigLoader()
        config = loader.load()
        
        assert "llama_cpp" in config.backends
        assert "localai" in config.backends
    
    def test_invalid_config_raises_error(self, temp_dir: Path) -> None:
        config_path = temp_dir / "config.yaml"
        config_path.write_text("invalid: [unclosed")
        
        loader = ConfigLoader()
        with pytest.raises(ConfigError):
            loader.load(config_path=config_path)
    
    def test_generate_default_config(self) -> None:
        loader = ConfigLoader()
        yaml_content = loader.generate_default_config()
        
        assert "source_dir:" in yaml_content
        assert "backends:" in yaml_content
        assert "llama_cpp:" in yaml_content
        assert "localai:" in yaml_content
