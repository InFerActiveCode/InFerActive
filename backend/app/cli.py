#!/usr/bin/env python3
"""Command line interface for InferActive server."""

import click
import os
import uvicorn
from pathlib import Path
from typing import Optional


@click.command()
@click.option(
    "--backend",
    default="vllm",
    type=click.Choice(["vllm"], case_sensitive=False),
    help="Real inference backend to use when not serving a mock tree"
)
@click.option(
    "--model-path",
    required=False,
    type=click.Path(exists=True, path_type=Path),
    help="Path to the model directory"
)
@click.option(
    "--model-type",
    default="llama3",
    type=click.Choice(['llama', 'llama2', 'llama3', 'qwen', 'gemma', 'exaone'], case_sensitive=False),
    help="Type of model to load"
)
@click.option(
    "--mock-tree-path",
    required=False,
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    help="Path to a saved tree JSON file for mock serving"
)
@click.option(
    "--mock-initial-depth",
    default=5,
    type=int,
    help="Depth returned from the mock tree for generate_with_bfs"
)
@click.option(
    "--mock-expand-depth",
    default=4,
    type=int,
    help="Maximum subtree depth returned from the mock tree for explore_node"
)
@click.option(
    "--mock-postfix",
    default=" this is mockup sending",
    help="Synthetic continuation appended to mock leaves"
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
@click.option("--dtype", default="float16", help="Model dtype for vLLM")
@click.option(
    "--tensor-parallel-size",
    default=1,
    type=int,
    help="vLLM tensor parallel size"
)
@click.option(
    "--max-model-len",
    default=2048,
    type=int,
    help="Maximum vLLM model context length"
)
@click.option(
    "--gpu-memory-utilization",
    default=0.7,
    type=float,
    help="vLLM GPU memory utilization"
)
@click.option(
    "--max-logprobs",
    default=200,
    type=int,
    help="Maximum vLLM processed logprobs to request"
)
@click.option(
    "--enable-prefix-caching/--disable-prefix-caching",
    default=True,
    help="Enable or disable vLLM prefix caching"
)
@click.option(
    "--tree-max-depth",
    default=512,
    type=int,
    help="Maximum absolute token tree depth"
)
@click.option(
    "--tree-max-leaves",
    default=1000,
    type=int,
    help="Maximum retained tree leaves; 0 disables pruning"
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
    backend: str,
    model_path: Optional[Path],
    model_type: str,
    mock_tree_path: Optional[Path],
    mock_initial_depth: int,
    mock_expand_depth: int,
    mock_postfix: str,
    host: str,
    port: int,
    gpu_id: int,
    dtype: str,
    tensor_parallel_size: int,
    max_model_len: int,
    gpu_memory_utilization: float,
    max_logprobs: int,
    enable_prefix_caching: bool,
    tree_max_depth: int,
    tree_max_leaves: int,
    log_level: str,
    model_unload_timeout: int,
    batch_size: int,
    batch_timeout: float,
):
    """Start the InferActive inference server."""
    if model_path is None and mock_tree_path is None:
        raise click.UsageError("Provide either --model-path or --mock-tree-path.")

    if model_path is not None:
        os.environ["INFERACTIVE_MODEL_PATH"] = str(model_path)
    if mock_tree_path is not None:
        os.environ["INFERACTIVE_MOCK_TREE_PATH"] = str(mock_tree_path)
        os.environ["INFERACTIVE_MOCK_INITIAL_DEPTH"] = str(mock_initial_depth)
        os.environ["INFERACTIVE_MOCK_EXPAND_DEPTH"] = str(mock_expand_depth)
        os.environ["INFERACTIVE_MOCK_POSTFIX"] = mock_postfix
    os.environ["INFERACTIVE_BACKEND"] = backend.lower()
    os.environ["INFERACTIVE_MODEL_TYPE"] = model_type.lower()
    os.environ["INFERACTIVE_GPU_ID"] = str(gpu_id)
    os.environ["INFERACTIVE_DTYPE"] = dtype
    os.environ["INFERACTIVE_TENSOR_PARALLEL_SIZE"] = str(tensor_parallel_size)
    os.environ["INFERACTIVE_MAX_MODEL_LEN"] = str(max_model_len)
    os.environ["INFERACTIVE_GPU_MEMORY_UTILIZATION"] = str(gpu_memory_utilization)
    os.environ["INFERACTIVE_MAX_LOGPROBS"] = str(max_logprobs)
    os.environ["INFERACTIVE_ENABLE_PREFIX_CACHING"] = "true" if enable_prefix_caching else "false"
    os.environ["INFERACTIVE_TREE_MAX_DEPTH"] = str(tree_max_depth)
    os.environ["INFERACTIVE_TREE_MAX_LEAVES"] = str(tree_max_leaves)
    os.environ["INFERACTIVE_LOG_LEVEL"] = log_level.upper()
    os.environ["INFERACTIVE_MODEL_UNLOAD_TIMEOUT"] = str(model_unload_timeout)
    os.environ["INFERACTIVE_BATCH_SIZE"] = str(batch_size)
    os.environ["INFERACTIVE_BATCH_TIMEOUT"] = str(batch_timeout)

    if model_path is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)

    click.echo("Starting InferActive server...")
    if mock_tree_path is not None:
        click.echo(f"Mock tree: {mock_tree_path}")
        click.echo(f"Mock depth: initial={mock_initial_depth}, expand={mock_expand_depth}")
    else:
        click.echo(f"Model: {model_path} (type: {model_type})")
        click.echo(f"Backend: {backend}")
        if backend.lower() == "vllm":
            click.echo(
                "vLLM: "
                f"tp={tensor_parallel_size}, max_model_len={max_model_len}, "
                f"max_logprobs={max_logprobs}, prefix_caching={enable_prefix_caching}"
            )
    click.echo(f"Server: {host}:{port}")
    if model_path is not None:
        click.echo(f"GPU: {gpu_id}")

    from app.main import app

    uvicorn.run(
        app,
        host=host,
        port=port,
        log_level=log_level.lower(),
        access_log=log_level.lower() == "debug",
    )


if __name__ == "__main__":
    main()
