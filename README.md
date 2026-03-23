# Link Models

Cross-platform model linker for LLM inference engines. Automatically synchronizes GGUF model files across multiple backends including llama.cpp, LocalAI, LM Studio, Ollama, and more.

## Features

- **Multi-Backend Support**: Sync to llama.cpp, LocalAI, LM Studio, Ollama, TextGen WebUI, GPT4All, KoboldCpp, and vLLM simultaneously
- **Add-Only Mode**: Add models to backends without deleting (prevents polluting your model selection)
- **Per-Backend Ignore Files**: Exclude specific models from specific backends using `.linkmodelsignore`
- **Bidirectional Sync**: Source directory is ground truth; changes propagate everywhere
- **Download Detection**: Automatically waits for partial downloads (.part, .tmp) to complete
- **Parallel GGUF Parsing**: Fast metadata extraction using multiprocessing
- **Cross-Platform**: Works on Linux, macOS, and Windows
- **Filesystem Watching**: Continuous monitoring mode for automatic synchronization
- **Service Installation**: Easy systemd/launchd service setup
- **Single Binary**: Can be compiled to a standalone executable

## Installation

### Using UV (Recommended - Fastest)

With [uv](https://docs.astral.sh/uv/) installed, you can run `link-models` directly without installing:

```bash
# Run directly (no installation needed)
uvx link-models --help

# Or install as a tool
uv tool install link-models
```

### From PyPI (pip)

```bash
pip install link-models
```

### From Source (with uv)

```bash
git clone https://github.com/mudler/LocalAI
cd LocalAI/link_models_py
uv venv && uv pip install -e ".[dev]"
```

### From Source (with pip)

```bash
git clone https://github.com/mudler/LocalAI
cd LocalAI/link_models_py
pip install -e ".[dev]"
```

### Standalone Binary

Download pre-built binaries from the [releases page](https://github.com/mudler/LocalAI/releases).

## Quick Start

### 1. Generate Configuration

```bash
link-models config --generate
```

This creates a `link_models.yaml` file with default settings.

### 2. Run One-Time Sync

```bash
# With default settings
link-models sync

# With custom directories
link-models sync --source /models --llama-cpp /llama_models --localai /localai_models
```

### 3. Run Filesystem Watcher

```bash
link-models watch
```

## Configuration

Configuration can be provided via:
1. Configuration file (`link_models.yaml`)
2. Environment variables
3. Command-line arguments

### Configuration File

```yaml
source_dir: /models

backends:
  llama_cpp:
    enabled: true
    output_dir: /llama_models
    generate_models_ini: true
    use_subdirs: true
  
  localai:
    enabled: true
    output_dir: /localai_models
    generate_yaml: true
    gpu_layers: -1
  
  lmstudio:
    enabled: false
    output_dir: /lmstudio_models
  
  ollama:
    enabled: false
    output_dir: /ollama_models
    generate_modelfile: true
  
  textgen:
    enabled: false
    output_dir: /textgen_models
    generate_model_configs: true
  
  gpt4all:
    enabled: false
    output_dir: /gpt4all_models
    generate_config: true
  
  koboldcpp:
    enabled: false
    output_dir: /koboldcpp_models
    generate_kcpps: true
    default_context_size: 4096
  
  vllm:
    enabled: false
    output_dir: /vllm_models
    generate_config: true

watch:
  enabled: false
  check_interval: 2.0
  stable_count: 3

logging:
  level: INFO
  json_format: false

sync:
  dry_run: false
  prefer_hardlinks: true
  add_only: false  # Set to true to never delete from backends
  preserve_orphans: false
  global_ignore_file: null  # Optional: path to global ignore file
```

### Environment Variables

Environment variables use the prefix `LINK_MODELS_` with double underscores for nesting:

```bash
export LINK_MODELS_SOURCE_DIR=/models
export LINK_MODELS_BACKENDS__LLAMA_CPP__OUTPUT_DIR=/llama_models
export LINK_MODELS_WATCH__ENABLED=true
```

## Add-Only Mode

Add-only mode prevents deletion from backends. This is useful when:

- You want to test models in some backends without affecting others
- You have models in backends that shouldn't be automatically removed
- You want to gradually add models to new backends without cleanup

### Enabling Add-Only Mode

```yaml
sync:
  add_only: true
```

Or via CLI:

```bash
link-models sync --add-only
```

Or via environment variable:

```bash
export LINK_MODELS_SYNC__ADD_ONLY=true
```

### Behavior in Add-Only Mode

- **New models**: Added to all enabled backends
- **Updated models**: Updated in place in all backends  
- **Deleted models from source**: NOT removed from backends (preserved)
- **Deleted from backend**: NOT restored from source
- **Orphan cleanup**: SKIPPED entirely

## Ignore Files

Ignore files prevent specific models from being synced to specific backends. This is useful when:

- A model doesn't work well with a particular backend
- You only want certain models in certain backends
- You want to test models silently without affecting other backends

### Global Ignore File

Apply to all backends:

```yaml
sync:
  global_ignore_file: /path/to/global-ignore.txt
```

Format (one glob pattern per line, lines starting with `#` are comments):

```
# Global ignore patterns
*small*
test-*
specific-model-name
```

### Per-Backend Ignore Files

Place a `.linkmodelsignore` file in any backend's output directory:

```bash
# In /ollama_models/.linkmodelsignore
model-a
model-b-*

# This will prevent model-a and any model starting with model-b- 
# from being added to the Ollama backend
```

Or configure explicitly in backend:

```yaml
backends:
  ollama:
    enabled: true
    output_dir: /ollama_models
    ignore_file: /path/to/ollama-ignore.txt
```

### Ignore Pattern Examples

| Pattern | Matches |
|---------|---------|
| `*` | All models |
| `*-small` | Models ending with `-small` |
| `llama-*` | Models starting with `llama-` |
| `*q4_*` | Models containing `q4_` |
| `test*` | Models starting with `test` |

Patterns are case-insensitive.

## Supported Backends

### llama.cpp

Standard llama.cpp model storage with optional `models.ini` generation.

```yaml
backends:
  llama_cpp:
    enabled: true
    output_dir: /llama_models
    generate_models_ini: true
    use_subdirs: true
```

### LocalAI

LocalAI YAML configuration files per model.

```yaml
backends:
  localai:
    enabled: true
    output_dir: /localai_models
    generate_yaml: true
    yaml_prefix: "model-"
    gpu_layers: -1
    mmap: true
    f16: true
```

### LM Studio

LM Studio manifest files for UI integration.

```yaml
backends:
  lmstudio:
    enabled: true
    output_dir: /lmstudio_models
    generate_manifest: true
```

### Ollama

Ollama Modelfiles and manifest generation.

```yaml
backends:
  ollama:
    enabled: true
    output_dir: /ollama_models
    generate_modelfile: true
    additional_params:
      temperature: 0.7
      top_p: 0.9
```

### Text Generation WebUI (oobabooga)

Flat directory structure with optional settings.

```yaml
backends:
  textgen:
    enabled: true
    output_dir: /textgen_models
    generate_settings_yaml: false
    generate_model_configs: false
```

### GPT4All

Optional JSON config files for model settings.

```yaml
backends:
  gpt4all:
    enabled: true
    output_dir: /gpt4all_models
    generate_config: false
    default_context_size: 4096
    default_gpu_layers: -1
```

### KoboldCpp

`.kcpps` sidecar config files.

```yaml
backends:
  koboldcpp:
    enabled: true
    output_dir: /koboldcpp_models
    generate_kcpps: true
    default_context_size: 4096
    default_gpu_layers: -1
    default_threads: 5
```

### vLLM

HuggingFace-style directory structure with config.json.

```yaml
backends:
  vllm:
    enabled: true
    output_dir: /vllm_models
    generate_config: true
    trust_remote_code: true
    enforce_eager: false
```

## Multi-Backend Example

Here's a complete example syncing to multiple backends:

```yaml
source_dir: /models

sync:
  add_only: false
  prefer_hardlinks: true
  
backends:
  llama_cpp:
    enabled: true
    output_dir: /llama_models
    generate_models_ini: true
    use_subdirs: true
    ignore_file: /config/llama-ignore.txt
  
  localai:
    enabled: true
    output_dir: /localai_models
    generate_yaml: true
  
  ollama:
    enabled: true
    output_dir: /ollama_models
    generate_modelfile: true
  
  koboldcpp:
    enabled: true
    output_dir: /koboldcpp_models
    generate_kcpps: true
```

With an ignore file at `/config/llama-ignore.txt`:

```
# Don't sync small models to llama.cpp
*small*
# Don't sync test models anywhere
test-*
```

## Service Installation

### Linux (systemd)

```bash
# Install service
sudo link-models service install

# Start service
sudo systemctl start link-models

# Enable on boot
sudo systemctl enable link-models

# Check status
link-models service status
```

### macOS (launchd)

```bash
# Install and start service
link-models service install
launchctl start link-models
```

## Development

### Setup with UV (Recommended)

```bash
# Clone repository
git clone https://github.com/mudler/LocalAI
cd LocalAI/link_models_py

# Create uv environment and install dependencies
make uv-install-dev
# Or manually:
# uv venv && uv pip install -e ".[dev]"

# Run tests
make test

# Run linting
make lint

# Run type checking
make type-check

# Run the application
uv run link-models --help
```

### Setup with pip (Legacy)

```bash
# Clone repository
git clone https://github.com/mudler/LocalAI
cd LocalAI/link_models_py

# Install development dependencies
make install-dev

# Run tests
make test

# Run linting
make lint

# Run type checking
make type-check
```

### Building Standalone Executable

```bash
# Using PyInstaller (recommended)
make build

# Using Nuitka
make build-nuitka
```

## Project Structure

```
link_models_py/
├── src/link_models/
│   ├── __init__.py
│   ├── main.py              # CLI entry point
│   ├── backends/            # Backend implementations
│   │   ├── base.py
│   │   ├── llama_cpp.py
│   │   ├── localai.py
│   │   ├── lmstudio.py
│   │   ├── ollama.py
│   │   ├── textgen.py
│   │   ├── gpt4all.py
│   │   ├── koboldcpp.py
│   │   └── vllm.py
│   └── core/                # Core functionality
│       ├── config.py
│       ├── constants.py
│       ├── exceptions.py
│       ├── gguf_parser.py
│       ├── logging.py
│       ├── models.py
│       ├── service.py
│       ├── sync.py
│       └── watcher.py
├── tests/
│   ├── unit/               # Unit tests
│   └── integration/        # Integration tests
├── scripts/
│   └── build.py            # Build script
├── pyproject.toml
├── Makefile
└── README.md
```

## Architecture

### Core Components

- **ConfigLoader**: Loads and merges configuration from files, environment, and CLI
- **SyncEngine**: Orchestrates synchronization between source and backends
- **ModelFilter**: Handles per-backend and global ignore patterns
- **FileSystemWatcher**: Monitors directories for changes using watchdog
- **DownloadDetector**: Detects when file downloads are complete
- **ParallelGGUFParser**: Extracts metadata from GGUF files in parallel

### Backends

Each backend implements the `Backend` base class:

- **LlamaCppBackend**: Creates hardlinks and generates `models.ini`
- **LocalAIBackend**: Creates hardlinks and generates YAML configs
- **LMStudioBackend**: Creates hardlinks and generates manifest files
- **OllamaBackend**: Creates hardlinks and generates Modelfiles
- **TextGenBackend**: Creates hardlinks for TextGen WebUI
- **GPT4AllBackend**: Creates hardlinks with optional JSON configs
- **KoboldCppBackend**: Creates hardlinks and `.kcpps` sidecar files
- **vLLMBackend**: Creates hardlinks and HuggingFace-style config.json

### Synchronization Strategy

1. **Source is Ground Truth**: `/models` directory defines what should exist
2. **Add-Only Mode**: When enabled, never deletes from backends
3. **Ignore Patterns**: Per-backend and global filters prevent sync
4. **Bidirectional Propagation**: Changes in any directory sync to others
5. **Hardlink Preference**: Uses hardlinks when possible, falls back to symlinks
6. **Orphan Cleanup**: Files not in source are removed from destinations (skipped in add-only mode)

## Testing

The project includes comprehensive tests:

```bash
# Run all tests
make test

# Run only unit tests
make test-unit

# Run with coverage
make test-cov

# Run specific test file
pytest tests/unit/test_models.py -v
```

## License

MIT License - See [LICENSE](../LICENSE) for details.

## Contributing

Contributions are welcome! Please see our [Contributing Guide](../CONTRIBUTING.md) for details.

## Acknowledgments

- [llama.cpp](https://github.com/ggerganov/llama.cpp) - LLM inference in C/C++
- [LocalAI](https://github.com/mudler/LocalAI) - Self-hosted OpenAI API alternative
- [LM Studio](https://lmstudio.ai/) - Desktop app for LLMs
- [Ollama](https://ollama.com/) - Run LLMs locally
- [Text Generation WebUI](https://github.com/oobabooga/text-generation-webui) - Web UI for LLMs
- [GPT4All](https://gpt4all.io/) - Open-source LLMs
- [KoboldCpp](https://github.com/LostRuins/koboldcpp) - Easy AI text generation
- [vLLM](https://vllm.ai/) - High-performance LLM serving
- [GGUF](https://github.com/ggerganov/ggml/blob/master/docs/gguf.md) - GGUF file format specification
