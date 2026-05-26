"""FastAPI application for the InFerActive backend."""

import logging
import os
import sys
from contextlib import asynccontextmanager
from typing import Any, Optional

from fastapi import FastAPI, WebSocket
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

logger: Optional[logging.Logger] = None
connection_manager: Any = None
session_manager: Any = None
model_manager: Any = None
websocket_handler: Any = None
config: Any = None


def initialize_app() -> None:
    """Initialize config and managers after CLI/env setup."""
    global logger, connection_manager, session_manager, model_manager, websocket_handler, config

    if connection_manager is not None:
        return

    from api.connections import WebSocketConnectionManager, SessionManager
    from api.handlers import WebSocketMessageHandler
    from config.settings import get_config
    from utils.logging import setup_logging
    from utils.manager import ModelLifecycleManager

    setup_logging()
    logger = logging.getLogger(__name__)

    config = get_config()

    connection_manager = WebSocketConnectionManager()
    session_manager = SessionManager()
    model_manager = ModelLifecycleManager()

    websocket_handler = WebSocketMessageHandler(
        connection_manager=connection_manager,
        session_manager=session_manager,
        model_manager=model_manager
    )


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Handle application startup and shutdown."""
    initialize_app()

    logger.info("Starting InFerActive server")
    if config.mock_tree_path:
        logger.info("Mock tree: %s", config.mock_tree_path)
    else:
        logger.info("Model: %s (type: %s)", config.model_path, config.model_type)
    logger.info("Device: %s", config.device)
    logger.info("Batch size: %s, timeout: %ss", config.batch_size, config.batch_timeout)
    logger.info("Model unload timeout: %ss", config.model_unload_timeout)

    yield

    logger.info("Shutting down InFerActive server")
    await session_manager.cleanup_all_sessions()
    await model_manager.cleanup_model()

    logger.info("Server shutdown complete")


app = FastAPI(
    title="InferActive",
    description="Interactive token-level inference server for language models",
    version="0.1.0",
    lifespan=lifespan
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
async def root():
    """Root endpoint providing basic server information."""
    initialize_app()
    return {
        "name": "InferActive",
        "version": "0.1.0",
        "description": "Interactive token-level inference server for language models",
        "status": "running",
        "model": {
            "path": str(config.model_path),
            "backend": "mock" if config.mock_tree_path else config.backend,
            "type": config.model_type,
            "mock_tree_path": str(config.mock_tree_path) if config.mock_tree_path else None,
            "status": model_manager.current_status
        },
        "connections": {
            "active_connections": connection_manager.get_connection_count(),
            "active_sessions": session_manager.get_session_count()
        }
    }


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    initialize_app()
    return {
        "status": "healthy",
        "model_loaded": model_manager.is_ready,
        "active_connections": connection_manager.get_connection_count()
    }


@app.get("/stats")
async def get_server_stats():
    """Get detailed server statistics."""
    initialize_app()
    return {
        "connections": {
            "count": connection_manager.get_connection_count(),
            "ids": connection_manager.get_connection_ids()
        },
        "sessions": {
            "count": session_manager.get_session_count(),
            "ids": session_manager.get_session_ids()
        },
        "model": {
            "status": model_manager.current_status,
            "loaded": model_manager.is_ready,
            "loading": model_manager.is_loading,
            "path": str(config.model_path),
            "backend": "mock" if config.mock_tree_path else config.backend,
            "type": config.model_type,
            "mock_tree_path": str(config.mock_tree_path) if config.mock_tree_path else None
        },
        "config": {
            "batch_size": config.batch_size,
            "batch_timeout": config.batch_timeout,
            "model_unload_timeout": config.model_unload_timeout,
            "device": config.device
        },
        "vllm": {
            "dtype": config.dtype,
            "tensor_parallel_size": config.tensor_parallel_size,
            "max_model_len": config.max_model_len,
            "gpu_memory_utilization": config.gpu_memory_utilization,
            "max_logprobs": config.max_logprobs,
            "enable_prefix_caching": config.enable_prefix_caching,
            "tree_max_depth": config.tree_max_depth,
            "tree_max_leaves": config.tree_max_leaves
        }
    }


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    """Main WebSocket endpoint for inference communication."""
    initialize_app()
    await websocket_handler.handle_connection(websocket)


@app.exception_handler(Exception)
async def global_exception_handler(request, exc):
    """Global exception handler."""
    try:
        initialize_app()
        logger.error("Unhandled exception: %s", exc, exc_info=True)
        show_details = config.log_level == "DEBUG"
    except Exception:
        show_details = False

    return JSONResponse(
        status_code=500,
        content={
            "error": "Internal server error",
            "message": str(exc) if show_details else "An error occurred",
        },
    )


if __name__ == "__main__":
    import uvicorn

    if not os.environ.get("INFERACTIVE_MODEL_PATH") and not os.environ.get("INFERACTIVE_MOCK_TREE_PATH"):
        sys.stderr.write("Error: set INFERACTIVE_MODEL_PATH or INFERACTIVE_MOCK_TREE_PATH\n")
        sys.exit(1)

    initialize_app()

    uvicorn.run(
        "app.main:app",
        host=config.host,
        port=config.port,
        log_level=config.log_level.lower(),
        reload=False,
    )
