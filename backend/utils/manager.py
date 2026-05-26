"""Model lifecycle manager for loading and managing language models."""

import asyncio
import gc
import logging
import time
from typing import Any, Optional

from fastapi import WebSocket

from config.settings import get_config

try:
    import torch
except ImportError:
    torch = None

logger = logging.getLogger(__name__)


class ModelLifecycleManager:
    """Load, unload, and recover the active inference engine."""

    def __init__(self):
        self.config = get_config()
        self.inference_engine: Optional[Any] = None

        self.last_activity = time.time()
        self.unload_timer: Optional[asyncio.TimerHandle] = None
        self.is_loading = False
        self.load_lock = asyncio.Lock()

        mode = "mock" if self.config.mock_tree_path else self.config.backend
        logger.info(
            "ModelLifecycleManager initialized - mode=%s, path=%s, type=%s, device=%s",
            mode,
            self.config.mock_tree_path or self.config.model_path,
            self.config.model_type,
            self.config.device,
        )

    async def load_model(self, websocket: WebSocket, request_id: str) -> bool:
        """Load the configured inference engine if it is not already loaded."""
        async with self.load_lock:
            if self.inference_engine is not None:
                await self._send_model_status(websocket, request_id, "loaded", "Model already loaded")
                self._reset_unload_timer()
                return True
            
            if self.is_loading:
                await self._send_model_status(websocket, request_id, "loading", "Model is already being loaded")
                return False

            self.is_loading = True

            try:
                if self.config.mock_tree_path:
                    return await self._load_mock_tree(websocket, request_id)

                if self.config.backend != "vllm":
                    raise RuntimeError(f"Unsupported backend: {self.config.backend}")

                return await self._load_vllm_model(websocket, request_id)

            except Exception as e:
                self.is_loading = False
                error_msg = f"Failed to load model: {str(e)}"
                logger.error(error_msg, exc_info=True)

                await self._send_model_status(websocket, request_id, "error", error_msg)
            await self.cleanup_model()
            return False

    async def _load_vllm_model(self, websocket: WebSocket, request_id: str) -> bool:
        """Load the in-process vLLM BFS engine."""
        try:
            from inference.vllm_engine import VLLMInferenceEngine

            await self._send_loading_progress(websocket, request_id, 0, "Loading vLLM engine...")
            self.inference_engine = VLLMInferenceEngine(
                model_path=self.config.model_path,
                model_type=self.config.model_type,
                dtype=self.config.dtype,
                tensor_parallel_size=self.config.tensor_parallel_size,
                max_model_len=self.config.max_model_len,
                gpu_memory_utilization=self.config.gpu_memory_utilization,
                batch_size=self.config.batch_size,
                max_logprobs=self.config.max_logprobs,
                enable_prefix_caching=self.config.enable_prefix_caching,
                max_depth=self.config.tree_max_depth,
                max_leaves=self.config.tree_max_leaves,
            )
            await self.inference_engine.initialize()
            await self._send_model_status(
                websocket,
                request_id,
                "loaded",
                "vLLM engine loaded successfully",
                100,
            )

            self.is_loading = False
            self.last_activity = time.time()
            self._reset_unload_timer()
            return True
        except Exception as e:
            self.is_loading = False
            error_msg = f"Failed to load vLLM engine: {str(e)}"
            logger.error(error_msg, exc_info=True)
            await self._send_model_status(websocket, request_id, "error", error_msg)
            await self.cleanup_model()
            return False

    async def _load_mock_tree(self, websocket: WebSocket, request_id: str) -> bool:
        """Load the mock tree engine instead of a language model."""
        try:
            from inference.mock_engine import MockTreeInferenceEngine

            await self._send_loading_progress(websocket, request_id, 0, "Loading mock tree...")
            self.inference_engine = MockTreeInferenceEngine(
                tree_path=self.config.mock_tree_path,
                initial_depth=self.config.mock_initial_depth,
                expand_depth=self.config.mock_expand_depth,
                postfix=self.config.mock_postfix,
            )
            await self._send_model_status(
                websocket,
                request_id,
                "loaded",
                "Mock tree loaded successfully",
                100,
            )

            self.is_loading = False
            self.last_activity = time.time()
            self._reset_unload_timer()
            return True
        except Exception as e:
            self.is_loading = False
            error_msg = f"Failed to load mock tree: {str(e)}"
            logger.error(error_msg, exc_info=True)
            await self._send_model_status(websocket, request_id, "error", error_msg)
            await self.cleanup_model()
            return False

    async def unload_model(self) -> None:
        """Unload the model due to inactivity."""
        async with self.load_lock:
            if self.inference_engine is None:
                return

            logger.info("Unloading model due to inactivity...")
            await self.cleanup_model()

    async def cleanup_model(self) -> None:
        """Clean up model resources and free GPU memory."""
        try:
            if self.inference_engine is not None:
                cleanup = getattr(self.inference_engine, "cleanup", None)
                if callable(cleanup):
                    cleanup()
                del self.inference_engine
                self.inference_engine = None
                logger.info("Inference engine cleaned up")

            gc.collect()

        except Exception as e:
            logger.error("Error during model cleanup: %s", e, exc_info=True)

    def _reset_unload_timer(self) -> None:
        """Reset the unload timer based on configured timeout."""
        if self.unload_timer:
            self.unload_timer.cancel()

        loop = asyncio.get_event_loop()
        self.unload_timer = loop.call_later(
            self.config.model_unload_timeout,
            lambda: asyncio.create_task(self.unload_model())
        )
        self.last_activity = time.time()
        logger.debug("Unload timer reset to %ss", self.config.model_unload_timeout)

    async def ensure_model_ready(self, websocket: WebSocket, request_id: str) -> bool:
        """Ensure the inference engine is loaded."""
        if self.inference_engine is None:
            return await self.load_model(websocket, request_id)

        self._reset_unload_timer()
        return True

    async def handle_inference_error(
        self,
        websocket: WebSocket,
        request_id: str,
        error: Exception
    ) -> bool:
        """Report inference errors and recover when possible."""
        error_msg = str(error)
        logger.error("Inference error: %s", error_msg, exc_info=True)

        critical_patterns = [
            "CUDA out of memory",
            "CUDA error",
            "device-side assert triggered",
            "illegal memory access"
        ]

        is_critical = any(pattern in error_msg for pattern in critical_patterns)

        if is_critical:
            logger.critical("Critical GPU error detected; cleaning up model state")

            await self._send_model_status(
                websocket,
                request_id,
                "critical_error",
                "Critical GPU error detected. Restart the backend process before continuing."
            )

            await self.cleanup_model()
            return False

        elif "out of memory" in error_msg.lower():
            await self._send_model_status(
                websocket,
                request_id,
                "error",
                "Memory error detected, cleaning up and reloading model..."
            )

            await self.cleanup_model()

            gc.collect()
            if torch is not None and torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.synchronize()

            return await self.load_model(websocket, request_id)

        await self._send_model_status(websocket, request_id, "error", f"Inference error: {error_msg}")
        return False

    @property
    def is_ready(self) -> bool:
        """Check if the model is currently loaded and ready."""
        return self.inference_engine is not None and not self.is_loading

    @property
    def current_status(self) -> str:
        """Get the current status of the model."""
        if self.is_loading:
            return "loading"
        if self.inference_engine is not None:
            return "loaded"
        return "unloaded"

    async def _send_model_status(
        self,
        websocket: WebSocket,
        request_id: str,
        status: str,
        message: str,
        progress: Optional[int] = None
    ) -> None:
        """Send status update via websocket."""
        payload = {
            "type": "model_status",
            "request_id": request_id,
            "status": status,
            "message": message
        }
        if progress is not None:
            payload["progress"] = progress

        await websocket.send_json(payload)

    async def _send_loading_progress(
        self,
        websocket: WebSocket,
        request_id: str,
        progress: int,
        message: str
    ) -> None:
        """Send loading progress update."""
        await websocket.send_json({
            "type": "loading_status",
            "request_id": request_id,
            "status": "loading",
            "progress": progress,
            "message": message
        })
