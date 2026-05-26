"""Logging utilities for InferActive server."""

import logging
import os
import sys
from typing import Optional


def setup_logging(log_level: Optional[str] = None) -> None:
    if log_level is None:
        log_level = os.environ.get("INFERACTIVE_LOG_LEVEL", "INFO")
    
    logging.basicConfig(
        level=getattr(logging, log_level.upper(), logging.INFO),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S',
        handlers=[
            logging.StreamHandler(sys.stdout)
        ]
    )
    
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
    
    logger = logging.getLogger('app')
    logger.info("Logging configured - Level: %s", log_level.upper())


def get_logger(name: str) -> logging.Logger:
    return logging.getLogger(name)
