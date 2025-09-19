"""Main FastAPI application for InferActive server."""

import asyncio
import logging
import sys
from contextlib import asynccontextmanager

from fastapi import FastAPI, WebSocket
from fastapi.middleware.cors import CORSMiddleware

# Don't import anything that uses config during module import
logger = None
connection_manager = None
session_manager = None  
model_manager = None
websocket_handler = None
config = None


def initialize_app():
    """Initialize all managers and config - called after env vars are set."""
    global logger, connection_manager, session_manager, model_manager, websocket_handler, config
    
    if connection_manager is not None:
        return  # Already initialized
    
    # Now safe to import and initialize everything
    from api.connections import WebSocketConnectionManager, SessionManager
    from api.handlers import WebSocketMessageHandler
    from config.settings import get_config
    from utils.manager import ModelLifecycleManager
    from utils.logging import setup_logging
    
    # Setup logging
    setup_logging()
    logger = logging.getLogger(__name__)
    
    # Get config
    config = get_config()
    
    # Create managers
    connection_manager = WebSocketConnectionManager()
    session_manager = SessionManager()
    model_manager = ModelLifecycleManager()
    
    # Create WebSocket handler
    websocket_handler = WebSocketMessageHandler(
        connection_manager=connection_manager,
        session_manager=session_manager,
        model_manager=model_manager
    )


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Application lifespan context manager.
    Handles startup and shutdown events.
    """
    # Initialize everything now that env vars are set
    initialize_app()
    
    logger.info(f"Starting InferActive server...")
    logger.info(f"Model: {config.model_path} (type: {config.model_type})")
    logger.info(f"Device: {config.device}")
    logger.info(f"Batch size: {config.batch_size}, timeout: {config.batch_timeout}s")
    logger.info(f"Model unload timeout: {config.model_unload_timeout}s")
    
    # Startup
    yield
    
    # Shutdown
    logger.info("Shutting down InferActive server...")
    
    # Clean up sessions
    await session_manager.cleanup_all_sessions()
    
    # Clean up model
    await model_manager.cleanup_model()
    
    logger.info("Server shutdown complete")


# Create FastAPI app
app = FastAPI(
    title="InferActive",
    description="Interactive token-level inference server for language models",
    version="0.1.0",
    lifespan=lifespan
)

# Add CORS middleware immediately (before app starts)
# Use permissive settings for development - can be configured later if needed
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all origins for development
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
async def root():
    """Root endpoint providing basic server information."""
    initialize_app()  # Ensure initialized
    return {
        "name": "InferActive",
        "version": "0.1.0",
        "description": "Interactive token-level inference server for language models",
        "status": "running",
        "model": {
            "path": str(config.model_path),
            "type": config.model_type,
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
    initialize_app()  # Ensure initialized
    return {
        "status": "healthy",
        "model_loaded": model_manager.is_ready,
        "active_connections": connection_manager.get_connection_count()
    }


@app.get("/stats")
async def get_server_stats():
    """Get detailed server statistics."""
    initialize_app()  # Ensure initialized
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
            "type": config.model_type
        },
        "config": {
            "batch_size": config.batch_size,
            "batch_timeout": config.batch_timeout,
            "model_unload_timeout": config.model_unload_timeout,
            "device": config.device
        }
    }


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    """Main WebSocket endpoint for inference communication."""
    initialize_app()  # Ensure initialized
    await websocket_handler.handle_connection(websocket)


# Error handlers
@app.exception_handler(Exception)
async def global_exception_handler(request, exc):
    """Global exception handler."""
    try:
        initialize_app()
        logger.error(f"Unhandled exception: {exc}", exc_info=True)
        show_details = config.log_level == "DEBUG"
    except:
        show_details = False
    
    return {
        "error": "Internal server error",
        "message": str(exc) if show_details else "An error occurred"
    }


if __name__ == "__main__":
    # This allows the module to be run directly for development
    import uvicorn
    import os
    
    if not os.environ.get("INFERACTIVE_MODEL_PATH"):
        print("Error: INFERACTIVE_MODEL_PATH environment variable is required")
        sys.exit(1)
    
    initialize_app()
    
    uvicorn.run(
        "app.main:app",
        host=config.host,
        port=config.port,
        log_level=config.log_level.lower(),
        reload=False  # Disable reload in production
    )