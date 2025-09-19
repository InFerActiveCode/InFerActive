"""Logging utilities for InferActive server."""

import logging
import sys
import os
from typing import Optional


def setup_logging(log_level: Optional[str] = None) -> None:
    """
    Configure logging for the application.
    
    Args:
        log_level: Override log level (if None, uses environment variable or INFO).
    """
    if log_level is None:
        # Get log level from environment variable, don't use config
        log_level = os.environ.get("INFERACTIVE_LOG_LEVEL", "INFO")
    
    # Configure root logger
    logging.basicConfig(
        level=getattr(logging, log_level.upper(), logging.INFO),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S',
        handlers=[
            logging.StreamHandler(sys.stdout)
        ]
    )
    
    # Set specific loggers to appropriate levels
    loggers_config = {
        'app': log_level.upper(),
        'models': log_level.upper(),
        'inference': log_level.upper(),
        'api': log_level.upper(),
        'config': log_level.upper(),
        'utils': log_level.upper(),
        'uvicorn.access': 'WARNING' if log_level.upper() != 'DEBUG' else 'INFO',
        'uvicorn.error': 'INFO',
        'transformers': 'WARNING',
        'torch': 'WARNING',
        'asyncio': 'WARNING'
    }
    
    for logger_name, level in loggers_config.items():
        logging.getLogger(logger_name).setLevel(getattr(logging, level))
    
    # Create application logger
    logger = logging.getLogger('app')
    logger.info(f"Logging configured - Level: {log_level.upper()}")


def get_logger(name: str) -> logging.Logger:
    """
    Get a logger with the specified name.
    
    Args:
        name: Name for the logger (usually module name).
        
    Returns:
        logging.Logger: Configured logger instance.
    """
    return logging.getLogger(name)