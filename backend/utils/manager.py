
"""Model lifecycle manager for loading and managing language models."""

import asyncio
import gc
import logging
import os
import sys
import time
from typing import Optional

import torch
from fastapi import WebSocket
from transformers import AutoModelForCausalLM, AutoTokenizer

from config.settings import get_config
from inference.engine import TokenInferenceEngine

logger = logging.getLogger(__name__)


class ModelLifecycleManager:
    """
    Manages the lifecycle of a language model including loading, unloading,
    and automatic cleanup based on inactivity.
    """
    
    def __init__(self):
        """Initialize the model manager with configuration."""
        self.config = get_config()
        self.model: Optional[AutoModelForCausalLM] = None
        self.tokenizer: Optional[AutoTokenizer] = None
        self.inference_engine: Optional[TokenInferenceEngine] = None
        
        self.last_activity = time.time()
        self.unload_timer: Optional[asyncio.TimerHandle] = None
        self.is_loading = False
        self.load_lock = asyncio.Lock()
        
        logger.info(
            f"ModelLifecycleManager initialized - Path: {self.config.model_path}, "
            f"Type: {self.config.model_type}, Device: {self.config.device}"
        )
    
    async def load_model(self, websocket: WebSocket, request_id: str) -> bool:
        """
        Load the model and tokenizer if not already loaded.
        
        Args:
            websocket: WebSocket connection for status updates.
            request_id: Request ID for tracking.
            
        Returns:
            bool: True if model was loaded successfully, False otherwise.
        """
        async with self.load_lock:
            if self.model is not None:
                await self._send_model_status(websocket, request_id, "loaded", "Model already loaded")
                self._reset_unload_timer()
                return True
            
            if self.is_loading:
                await self._send_model_status(websocket, request_id, "loading", "Model is already being loaded")
                return False
            
            self.is_loading = True
            
            try:
                await self._send_loading_progress(websocket, request_id, 0, "Loading tokenizer...")
                
                # Load tokenizer
                self.tokenizer = AutoTokenizer.from_pretrained(
                    str(self.config.model_path),
                    trust_remote_code=True
                )
                logger.info(f"Tokenizer loaded from {self.config.model_path}")
                
                await self._send_loading_progress(websocket, request_id, 30, "Loading model weights...")
                
                # Load model
                self.model = AutoModelForCausalLM.from_pretrained(
                    str(self.config.model_path),
                    torch_dtype=torch.float16,
                    trust_remote_code=True,
                    device_map=self.config.device if torch.cuda.is_available() else "cpu"
                )
                self.model.eval()
                logger.info(f"Model loaded from {self.config.model_path}")
                
                await self._send_loading_progress(websocket, request_id, 90, "Initializing inference engine...")
                
                # Initialize inference engine
                self.inference_engine = TokenInferenceEngine(
                    model=self.model,
                    tokenizer=self.tokenizer,
                    model_type=self.config.model_type,
                    device=self.config.device,
                    batch_size=self.config.batch_size,
                    batch_timeout=self.config.batch_timeout
                )
                logger.info("TokenInferenceEngine initialized")
                
                await self._send_model_status(websocket, request_id, "loaded", "Model loaded successfully", 100)
                
                self.is_loading = False
                self.last_activity = time.time()
                self._reset_unload_timer()
                return True
                
            except Exception as e:
                self.is_loading = False
                error_msg = f"Failed to load model: {str(e)}"
                logger.error(error_msg, exc_info=True)
                
                await self._send_model_status(websocket, request_id, "error", error_msg)
                await self.cleanup_model()
                return False
    
    async def unload_model(self) -> None:
        """Unload the model due to inactivity."""
        async with self.load_lock:
            if self.model is None:
                return
            
            logger.info("Unloading model due to inactivity...")
            await self.cleanup_model()
    
    async def cleanup_model(self) -> None:
        """Clean up model resources and free GPU memory."""
        try:
            if self.model is not None:
                del self.model
                self.model = None
                logger.info("Model cleaned up")
            
            if self.tokenizer is not None:
                del self.tokenizer
                self.tokenizer = None
                logger.info("Tokenizer cleaned up")
            
            if self.inference_engine is not None:
                del self.inference_engine
                self.inference_engine = None
                logger.info("Inference engine cleaned up")
            
            # Force garbage collection and clear CUDA cache
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
                logger.info("GPU memory cleared")
                
        except Exception as e:
            logger.error(f"Error during model cleanup: {e}", exc_info=True)
    
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
        logger.debug(f"Unload timer reset to {self.config.model_unload_timeout}s")
    
    async def ensure_model_ready(self, websocket: WebSocket, request_id: str) -> bool:
        """
        Ensure the model is loaded and ready for inference.
        
        Args:
            websocket: WebSocket connection for status updates.
            request_id: Request ID for tracking.
            
        Returns:
            bool: True if model is ready for inference.
        """
        if self.model is None:
            return await self.load_model(websocket, request_id)
        
        self._reset_unload_timer()
        return True
    
    async def handle_inference_error(
        self, 
        websocket: WebSocket, 
        request_id: str, 
        error: Exception
    ) -> bool:
        """
        Handle inference errors and attempt recovery.
        
        Args:
            websocket: WebSocket connection for error reporting.
            request_id: Request ID for tracking.
            error: The error that occurred.
            
        Returns:
            bool: True if recovery was attempted successfully.
        """
        error_msg = str(error)
        logger.error(f"Inference error: {error_msg}", exc_info=True)
        
        # Check for critical errors that require restart
        critical_patterns = [
            "CUDA out of memory",
            "CUDA error",
            "device-side assert triggered", 
            "illegal memory access"
        ]
        
        is_critical = any(pattern in error_msg for pattern in critical_patterns)
        
        if is_critical:
            logger.critical("Critical GPU error detected! Initiating cleanup and restart...")
            
            await self._send_model_status(
                websocket, 
                request_id, 
                "critical_error",
                "Critical error detected. System will restart in 5 seconds..."
            )
            
            # Cleanup model
            await self.cleanup_model()
            
            # Wait for clients to receive message
            await asyncio.sleep(5)
            
            # Force restart
            logger.info("Restarting server...")
            os.execv(sys.executable, [sys.executable] + sys.argv)
        
        # Handle memory errors with recovery
        elif "out of memory" in error_msg.lower():
            await self._send_model_status(
                websocket,
                request_id, 
                "error",
                "Memory error detected, cleaning up and reloading model..."
            )
            
            # Cleanup and attempt reload
            await self.cleanup_model()
            
            # Force garbage collection
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
            
            # Try to reload
            return await self.load_model(websocket, request_id)
        
        # For other errors, just report
        await self._send_model_status(websocket, request_id, "error", f"Inference error: {error_msg}")
        return False
    
    @property
    def is_ready(self) -> bool:
        """Check if the model is currently loaded and ready."""
        return self.model is not None and not self.is_loading
    
    @property
    def current_status(self) -> str:
        """Get the current status of the model."""
        if self.is_loading:
            return "loading"
        elif self.model is not None:
            return "loaded"
        else:
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