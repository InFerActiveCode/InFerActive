"""
InferActive: Interactive token-level inference server for language models.

This package provides a FastAPI-based server that enables real-time, 
token-level inference with tree exploration and advanced sampling methods.
"""

__version__ = "0.1.0"
__author__ = "InferActive Contributors"

from config.settings import get_config, set_config
from utils.token_node import TokenNode

__all__ = [
    "get_config",
    "set_config", 
    "TokenNode",
    "__version__"
]