# Link Models

Cross-platform model linker for LLM inference engines. Automatically synchronizes GGUF model files across multiple backends including llama.cpp, LocalAI, and LM Studio.

## Features

- **Multi-Backend Support**: Sync to llama.cpp, LocalAI, and LM Studio simultaneously
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

watch:
  enabled: false
  check_interval: 2.0
  stable_count: 3

logging:
  level: INFO
  json_format: false
```

### Environment Variables

Environment variables use the prefix `LINK_MODELS_` with double underscores for nesting:

```bash
export LINK_MODELS_SOURCE_DIR=/models
export LINK_MODELS_BACKENDS__LLAMA_CPP__OUTPUT_DIR=/llama_models
export LINK_MODELS_WATCH__ENABLED=true
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
│   │   └── lmstudio.py
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
- **FileSystemWatcher**: Monitors directories for changes using watchdog
- **DownloadDetector**: Detects when file downloads are complete
- **ParallelGGUFParser**: Extracts metadata from GGUF files in parallel

### Backends

Each backend implements the `Backend` base class:

- **LlamaCppBackend**: Creates hardlinks and generates `models.ini`
- **LocalAIBackend**: Creates hardlinks and generates YAML configs
- **LMStudioBackend**: Creates hardlinks and generates manifest files

### Synchronization Strategy

1. **Source is Ground Truth**: `/models` directory defines what should exist
2. **Bidirectional Propagation**: Changes in any directory sync to others
3. **Hardlink Preference**: Uses hardlinks when possible, falls back to symlinks
4. **Orphan Cleanup**: Files not in source are removed from destinations

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
- [GGUF](https://github.com/ggerganov/ggml/blob/master/docs/gguf.md) - GGUF file format specification
