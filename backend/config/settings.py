"""Configuration management for InferActive server."""

import os
from pathlib import Path
from typing import Literal, Optional, List
from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic import Field, field_validator


class ServerConfig(BaseSettings):
    """Configuration settings for InferActive server."""
    
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False
    )
    
    # Model configuration
    model_path: Path = Field(..., description="Path to the model directory")
    model_type: Literal["llama", "qwen", "exaone"] = Field(
        default="llama", 
        description="Type of model (llama, qwen, exaone)"
    )
    
    # Server configuration
    host: str = Field(default="0.0.0.0", description="Host to bind server to")
    port: int = Field(default=8008, description="Port to bind server to")
    
    # GPU configuration
    gpu_id: int = Field(default=0, description="GPU device ID")
    
    # Logging configuration
    log_level: str = Field(default="INFO", description="Logging level")
    
    # Model management
    model_unload_timeout: int = Field(
        default=600, 
        description="Time in seconds before unloading model due to inactivity"
    )
    
    # Batching configuration
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
    
    # CORS configuration - use sensible defaults
    cors_origins: List[str] = Field(
        default=["*"], 
        description="CORS allowed origins"
    )
    
    @field_validator("model_path")
    @classmethod
    def validate_model_path(cls, v):
        """Validate that model path exists."""
        path = Path(v)
        if not path.exists():
            raise ValueError(f"Model path does not exist: {path}")
        if not path.is_dir():
            raise ValueError(f"Model path must be a directory: {path}")
        return path
    
    @field_validator("model_type")
    @classmethod
    def validate_model_type(cls, v):
        """Ensure model type is lowercase."""
        return v.lower()
    
    @field_validator("log_level")
    @classmethod
    def validate_log_level(cls, v):
        """Ensure log level is uppercase."""
        return v.upper()
    
    @property
    def device(self) -> str:
        """Get the device string for PyTorch."""
        return "cuda:0" if os.environ.get("CUDA_VISIBLE_DEVICES") else "cpu"


# Global config instance
_config: Optional[ServerConfig] = None


def get_config() -> ServerConfig:
    """Get the global configuration instance."""
    global _config
    if _config is None:
        # Only read the essential environment variables, use defaults for the rest
        _config = ServerConfig(
            model_path=os.environ["INFERACTIVE_MODEL_PATH"],
            model_type=os.environ.get("INFERACTIVE_MODEL_TYPE", "llama"),
            host=os.environ.get("INFERACTIVE_HOST", "0.0.0.0"),
            port=int(os.environ.get("INFERACTIVE_PORT", "8008")),
            gpu_id=int(os.environ.get("INFERACTIVE_GPU_ID", "0")),
            log_level=os.environ.get("INFERACTIVE_LOG_LEVEL", "INFO"),
            model_unload_timeout=int(os.environ.get("INFERACTIVE_MODEL_UNLOAD_TIMEOUT", "600")),
            batch_size=int(os.environ.get("INFERACTIVE_BATCH_SIZE", "16")),
            batch_timeout=float(os.environ.get("INFERACTIVE_BATCH_TIMEOUT", "0.1"))
            # cors_origins will use the default ["*"] - don't read from env
        )
    return _config


def set_config(config: ServerConfig) -> None:
    """Set the global configuration instance."""
    global _config
    _config = config