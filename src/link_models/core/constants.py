"""Constants for link_models."""

from __future__ import annotations

# Default directory paths
DEFAULT_MODELS_SRC = "/models"
DEFAULT_MODELS_DST = "/llama_models"
DEFAULT_LOCALAI_DIR = "/localai_models"
DEFAULT_LMSTUDIO_DIR = "/lmstudio_models"
DEFAULT_OLLAMA_DIR = "/ollama_models"
DEFAULT_TEXTGEN_DIR = "/textgen_models"
DEFAULT_GPT4ALL_DIR = "/gpt4all_models"
DEFAULT_KOBOLDCPP_DIR = "/koboldcpp_models"
DEFAULT_VLLM_DIR = "/vllm_models"
DEFAULT_JAN_DIR = "/jan_models"
DEFAULT_LLAMA_CPP_PYTHON_DIR = "/llama-cpp-python_models"

# Service configuration
DEFAULT_SERVICE_NAME = "link-models"
DEFAULT_VENV_DIR = "/opt/link-models/venv"

# Download detection
DOWNLOAD_CHECK_INTERVAL = 2  # Seconds between size checks
DOWNLOAD_STABLE_COUNT = 3  # Consecutive stable checks to confirm complete
DOWNLOAD_MAX_WAIT = 3600  # Maximum seconds to wait for download
PARTIAL_DOWNLOAD_EXTENSIONS = frozenset(
    [
        ".part",
        ".tmp",
        ".temp",
        ".crdownload",
        ".download",
        "~",
    ]
)

# Model file patterns
GGUF_EXTENSION = ".gguf"
MMPROJ_PATTERN = "mmproj"

# Quantization preferences (in order of preference)
PREFERRED_QUANTIZATIONS = [
    "q4_k_m",
    "q5_k_m",
    "q8_0",
    "q6_k",
    "q5_k_s",
    "q4_k_s",
    "q3_k_m",
    "q2_k",
]

# Multipart patterns
MULTIPART_PATTERN = r"-(?P<part>\d+)-of-(?P<total>\d+)\.gguf$"

# GGUF metadata keys
GGUF_KEYS = {
    "architecture": "general.architecture",
    "name": "general.name",
    "file_type": "general.file_type",
    "context_length": [
        "{arch}.context_length",
        "llama.context_length",
        "gptneox.context_length",
        "general.context_length",
    ],
    "vocab_size": "llama.vocab_size",
    "block_count": "{arch}.block_count",
    "chat_template": "tokenizer.chat_template",
    "eos_token_id": "tokenizer.ggml.eos_token_id",
    "tokens": "tokenizer.ggml.tokens",
}

# Special EOS markers to detect
SPECIAL_EOS_MARKERS = [
    "<|eot_id|>",
    "<|im_end|>",
    "<|end|>",
    "<|endoftext|>",
    "<｜end▁of▁sentence｜>",
    "<|return|>",
    "<|eom_id|>",
    "</s>",
]

# Backend mappings
ARCHITECTURE_BACKENDS: dict[str, str] = {
    "llama": "llama-cpp",
    "llama2": "llama-cpp",
    "mistral": "llama-cpp",
    "mixtral": "llama-cpp",
    "qwen2": "llama-cpp",
    "phi3": "llama-cpp",
    "gemma": "llama-cpp",
    "qwen": "llama-cpp",
    "yi": "llama-cpp",
    "whisper": "whisper",
    "clip": "llama-cpp",
}

# Default backend for unknown architectures
DEFAULT_BACKEND = "llama-cpp"

# File permissions
DIR_PERMISSIONS = 0o755
FILE_PERMISSIONS = 0o644
