#!/usr/bin/env python3
"""Command line interface for InferActive server."""

import click
import uvicorn
import os
import sys
from pathlib import Path
from typing import Optional

@click.command()
@click.option(
    "--model-path", 
    required=True, 
    type=click.Path(exists=True, path_type=Path),
    help="Path to the model directory"
)
@click.option(
    "--model-type", 
    default="llama",
    type=click.Choice(['llama', 'qwen', 'exaone'], case_sensitive=False),
    help="Type of model to load (llama, qwen, or exaone)"
)
@click.option(
    "--host", 
    default="0.0.0.0", 
    help="Host to bind the server to"
)
@click.option(
    "--port", 
    default=8008, 
    type=int, 
    help="Port to bind the server to"
)
@click.option(
    "--gpu-id", 
    default=0, 
    type=int, 
    help="GPU device ID to use (default: 0)"
)
@click.option(
    "--log-level", 
    default="info",
    type=click.Choice(['debug', 'info', 'warning', 'error'], case_sensitive=False),
    help="Logging level"
)
@click.option(
    "--model-unload-timeout", 
    default=600, 
    type=int, 
    help="Time in seconds before unloading model due to inactivity"
)
@click.option(
    "--batch-size", 
    default=16, 
    type=int, 
    help="Maximum batch size for inference"
)
@click.option(
    "--batch-timeout", 
    default=0.1, 
    type=float, 
    help="Batch timeout in seconds"
)
def main(
    model_path: Path,
    model_type: str,
    host: str,
    port: int,
    gpu_id: int,
    log_level: str,
    model_unload_timeout: int,
    batch_size: int,
    batch_timeout: float
):
    """Start the InferActive inference server."""
    
    # Set environment variables for configuration
    os.environ["INFERACTIVE_MODEL_PATH"] = str(model_path)
    os.environ["INFERACTIVE_MODEL_TYPE"] = model_type.lower()
    os.environ["INFERACTIVE_GPU_ID"] = str(gpu_id)
    os.environ["INFERACTIVE_LOG_LEVEL"] = log_level.upper()
    os.environ["INFERACTIVE_MODEL_UNLOAD_TIMEOUT"] = str(model_unload_timeout)
    os.environ["INFERACTIVE_BATCH_SIZE"] = str(batch_size)
    os.environ["INFERACTIVE_BATCH_TIMEOUT"] = str(batch_timeout)
    
    # Set CUDA visible devices
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    
    click.echo(f"Starting InferActive server...")
    click.echo(f"Model: {model_path} (type: {model_type})")
    click.echo(f"Server: {host}:{port}")
    click.echo(f"GPU: {gpu_id}")
    
    # Import and start the FastAPI app
    from app.main import app
    
    uvicorn.run(
        app, 
        host=host, 
        port=port,
        log_level=log_level.lower(),
        access_log=log_level.lower() == "debug"
    )

if __name__ == "__main__":
    main()