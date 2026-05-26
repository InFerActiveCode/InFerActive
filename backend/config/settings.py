"""Configuration management for InferActive server."""

import os
from pathlib import Path
from typing import List, Literal, Optional

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class ServerConfig(BaseSettings):
    """Configuration settings for InferActive server."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False
    )

    model_path: Path = Field(default=Path("."), description="Path to the model directory")
    backend: Literal["vllm"] = Field(
        default="vllm",
        description="Real inference backend to use when mock_tree_path is not set"
    )
    model_type: Literal["llama", "llama2", "llama3", "qwen", "gemma", "exaone"] = Field(
        default="llama3",
        description="Type of model (llama, llama2, llama3, qwen, gemma, exaone)"
    )
    dtype: str = Field(default="float16", description="Model dtype for vLLM")
    tensor_parallel_size: int = Field(
        default=1,
        ge=1,
        description="vLLM tensor parallel size"
    )
    max_model_len: int = Field(
        default=2048,
        ge=1,
        description="Maximum vLLM model context length"
    )
    gpu_memory_utilization: float = Field(
        default=0.7,
        gt=0.0,
        le=1.0,
        description="vLLM GPU memory utilization"
    )
    max_logprobs: int = Field(
        default=200,
        description="Maximum vLLM processed logprobs to request"
    )
    enable_prefix_caching: bool = Field(
        default=True,
        description="Enable vLLM prefix caching"
    )
    tree_max_depth: int = Field(
        default=512,
        ge=1,
        description="Maximum absolute token tree depth"
    )
    tree_max_leaves: int = Field(
        default=1000,
        ge=0,
        description="Maximum retained tree leaves; 0 disables pruning"
    )
    mock_tree_path: Optional[Path] = Field(
        default=None,
        description="Optional saved tree JSON used instead of loading a model"
    )
    mock_initial_depth: int = Field(
        default=5,
        ge=0,
        description="Depth returned from the mock tree for generate_with_bfs"
    )
    mock_expand_depth: int = Field(
        default=4,
        ge=0,
        description="Maximum subtree depth returned by the mock tree for explore_node"
    )
    mock_postfix: str = Field(
        default=" this is mockup sending",
        description="Synthetic continuation appended to mock leaves"
    )

    host: str = Field(default="0.0.0.0", description="Host to bind server to")
    port: int = Field(default=8008, description="Port to bind server to")

    gpu_id: int = Field(default=0, description="GPU device ID")

    log_level: str = Field(default="INFO", description="Logging level")

    model_unload_timeout: int = Field(
        default=600,
        description="Time in seconds before unloading model due to inactivity"
    )

    batch_size: int = Field(
        default=16,
        ge=1,
        le=128,
        description="Maximum batch size for inference"
    )
    batch_timeout: float = Field(
        default=0.1,
        gt=0.0,
        description="Batch timeout in seconds"
    )

    cors_origins: List[str] = Field(
        default=["*"],
        description="CORS allowed origins"
    )

    @field_validator("model_path")
    @classmethod
    def validate_model_path(cls, v):
        path = Path(v)
        if not path.exists():
            raise ValueError(f"Model path does not exist: {path}")
        if not path.is_dir():
            raise ValueError(f"Model path must be a directory: {path}")
        return path

    @field_validator("mock_tree_path")
    @classmethod
    def validate_mock_tree_path(cls, v):
        if v is None:
            return None
        path = Path(v)
        if not path.exists():
            raise ValueError(f"Mock tree path does not exist: {path}")
        if not path.is_file():
            raise ValueError(f"Mock tree path must be a JSON file: {path}")
        return path

    @field_validator("model_type", mode="before")
    @classmethod
    def validate_model_type(cls, v):
        return str(v).lower()

    @field_validator("backend", mode="before")
    @classmethod
    def validate_backend(cls, v):
        return str(v).lower()

    @field_validator("log_level", mode="before")
    @classmethod
    def validate_log_level(cls, v):
        return str(v).upper()

    @property
    def device(self) -> str:
        return "cuda:0" if os.environ.get("CUDA_VISIBLE_DEVICES") else "cpu"


_config: Optional[ServerConfig] = None


def get_config() -> ServerConfig:
    global _config
    if _config is None:
        model_path = os.environ.get("INFERACTIVE_MODEL_PATH")
        mock_tree_path = os.environ.get("INFERACTIVE_MOCK_TREE_PATH")
        if not model_path and not mock_tree_path:
            raise RuntimeError(
                "Set INFERACTIVE_MODEL_PATH for real inference or "
                "INFERACTIVE_MOCK_TREE_PATH for mock tree serving."
            )

        _config = ServerConfig(
            model_path=model_path or ".",
            backend=os.environ.get("INFERACTIVE_BACKEND", "vllm"),
            model_type=os.environ.get("INFERACTIVE_MODEL_TYPE", "llama3"),
            dtype=os.environ.get("INFERACTIVE_DTYPE", "float16"),
            tensor_parallel_size=int(os.environ.get("INFERACTIVE_TENSOR_PARALLEL_SIZE", "1")),
            max_model_len=int(os.environ.get("INFERACTIVE_MAX_MODEL_LEN", "2048")),
            gpu_memory_utilization=float(os.environ.get("INFERACTIVE_GPU_MEMORY_UTILIZATION", "0.7")),
            max_logprobs=int(os.environ.get("INFERACTIVE_MAX_LOGPROBS", "200")),
            enable_prefix_caching=os.environ.get("INFERACTIVE_ENABLE_PREFIX_CACHING", "true").lower() in {"1", "true", "yes", "on"},
            tree_max_depth=int(os.environ.get("INFERACTIVE_TREE_MAX_DEPTH", "512")),
            tree_max_leaves=int(os.environ.get("INFERACTIVE_TREE_MAX_LEAVES", "1000")),
            mock_tree_path=mock_tree_path,
            mock_initial_depth=int(os.environ.get("INFERACTIVE_MOCK_INITIAL_DEPTH", "5")),
            mock_expand_depth=int(os.environ.get("INFERACTIVE_MOCK_EXPAND_DEPTH", "4")),
            mock_postfix=os.environ.get("INFERACTIVE_MOCK_POSTFIX", " this is mockup sending"),
            host=os.environ.get("INFERACTIVE_HOST", "0.0.0.0"),
            port=int(os.environ.get("INFERACTIVE_PORT", "8008")),
            gpu_id=int(os.environ.get("INFERACTIVE_GPU_ID", "0")),
            log_level=os.environ.get("INFERACTIVE_LOG_LEVEL", "INFO"),
            model_unload_timeout=int(os.environ.get("INFERACTIVE_MODEL_UNLOAD_TIMEOUT", "600")),
            batch_size=int(os.environ.get("INFERACTIVE_BATCH_SIZE", "16")),
            batch_timeout=float(os.environ.get("INFERACTIVE_BATCH_TIMEOUT", "0.1"))
        )
    return _config


def set_config(config: ServerConfig) -> None:
    global _config
    _config = config
