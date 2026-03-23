"""Backend implementations for different LLM inference engines."""

from .base import Backend, BackendResult, LinkResult
from .llama_cpp import LlamaCppBackend
from .localai import LocalAIBackend
from .lmstudio import LMStudioBackend
from .ollama import OllamaBackend
from .textgen import TextGenBackend
from .gpt4all import GPT4AllBackend
from .koboldcpp import KoboldCppBackend
from .vllm import vLLMBackend
from .jan import JanBackend
from .llama_cpp_python import LlamaCppPythonBackend

__all__ = [
    "Backend",
    "BackendResult",
    "LinkResult",
    "LlamaCppBackend",
    "LocalAIBackend",
    "LMStudioBackend",
    "OllamaBackend",
    "TextGenBackend",
    "GPT4AllBackend",
    "KoboldCppBackend",
    "vLLMBackend",
    "JanBackend",
    "LlamaCppPythonBackend",
]
