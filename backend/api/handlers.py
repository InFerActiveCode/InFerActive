"""WebSocket message handlers and routing."""

import asyncio
import logging
import os
from typing import Any, Dict

from fastapi import WebSocket, WebSocketDisconnect

from api.connections import WebSocketConnectionManager, SessionManager, SessionState
from utils.manager import ModelLifecycleManager
from utils.token_node import TokenNode

logger = logging.getLogger(__name__)


class WebSocketMessageHandler:
    """Handles WebSocket messages and routing."""

    def __init__(
        self,
        connection_manager: WebSocketConnectionManager,
        session_manager: SessionManager,
        model_manager: ModelLifecycleManager
    ):
        self.connection_manager = connection_manager
        self.session_manager = session_manager
        self.model_manager = model_manager

    async def handle_connection(self, websocket: WebSocket) -> None:
        """Handle a WebSocket connection lifecycle."""
        connection_id = await self.connection_manager.connect(websocket)
        session = await self.session_manager.create_session(connection_id, self.connection_manager)

        try:
            while True:
                data = await websocket.receive_json()
                await self.route_message(websocket, session, data)

        except WebSocketDisconnect:
            logger.info("WebSocket disconnected: %s", connection_id)
        except Exception as e:
            logger.exception("WebSocket error in session %s: %s", connection_id, e)
            try:
                await websocket.send_json({
                    "type": "error",
                    "message": str(e)
                })
            except Exception:
                pass
        finally:
            await self.session_manager.remove_session(connection_id)
            self.connection_manager.disconnect(connection_id)

    async def route_message(
        self,
        websocket: WebSocket,
        session: SessionState,
        data: Dict[str, Any]
    ) -> None:
        """Route an incoming WebSocket message."""
        message_type = data.get("type")
        request_id = data.get("request_id", os.urandom(4).hex())

        logger.debug("Handling message type %s (request_id=%s)", message_type, request_id)

        try:
            if message_type == "check_model_status":
                await self._handle_check_model_status(websocket, request_id)

            elif message_type == "load_model":
                await self._handle_load_model(websocket, request_id)

            elif message_type == "explore_node":
                await self._handle_explore_node(websocket, session, data)

            elif message_type == "generate_with_bfs":
                await self._handle_generate_with_bfs(websocket, session, data)

            elif message_type == "generate_with_smc":
                logger.info("Handling legacy generate_with_smc request as generate_with_bfs")
                await self._handle_generate_with_bfs(websocket, session, data)

            else:
                logger.warning("Unknown message type: %s", message_type)
                await websocket.send_json({
                    "type": "error",
                    "request_id": request_id,
                    "message": f"Unknown message type: {message_type}"
                })

        except Exception as e:
            logger.exception("Error handling message %s: %s", message_type, e)
            await websocket.send_json({
                "type": "error",
                "request_id": request_id,
                "message": f"Error processing {message_type}: {str(e)}"
            })

    async def _handle_check_model_status(
        self,
        websocket: WebSocket,
        request_id: str
    ) -> None:
        """Handle model status check request."""
        status = self.model_manager.current_status
        await websocket.send_json({
            "type": "model_status",
            "request_id": request_id,
            "status": status,
            "message": f"Model is {status}"
        })

    async def _handle_load_model(
        self,
        websocket: WebSocket,
        request_id: str
    ) -> None:
        """Handle model loading request."""
        await self.model_manager.load_model(websocket, request_id)

    async def _handle_explore_node(
        self,
        websocket: WebSocket,
        session: SessionState,
        data: Dict[str, Any]
    ) -> None:
        """Handle node exploration request."""
        request_id = data.get("request_id", os.urandom(4).hex())

        if not await self.model_manager.ensure_model_ready(websocket, request_id):
            return

        node = None
        if request_id in session.tree_storage:
            node = session.tree_storage[request_id]
            if data.get("node_id") is not None:
                node = node.get_node(data["node_id"])

        if node is None:
            await websocket.send_json({
                "type": "error",
                "request_id": request_id,
                "message": "Node not found"
            })
            return

        try:
            k = data.get("k", 5)
            temperature = data.get("temperature", 0.7)
            top_p = data.get("top_p", 0.9)
            min_p = data.get("min_p", 0.05)
            depth_to_explore = data.get("depth", 3)
            if getattr(self.model_manager.inference_engine, "is_mock", False):
                depth_to_explore = data.get("depth", data.get("max_tokens", depth_to_explore))

            self.model_manager.inference_engine.set_broadcast_queue(session.broadcast_queue)

            asyncio.create_task(
                self.model_manager.inference_engine.explore_node(
                    node, depth_to_explore, k, temperature, top_p, min_p, extend_greedy=True
                )
            )

        except Exception as e:
            await self.model_manager.handle_inference_error(websocket, request_id, e)

    async def _handle_generate_with_bfs(
        self,
        websocket: WebSocket,
        session: SessionState,
        data: Dict[str, Any]
    ) -> None:
        """Handle breadth-first tree generation request."""
        request_id = data.get("request_id", os.urandom(4).hex())

        if not await self.model_manager.ensure_model_ready(websocket, request_id):
            return

        try:
            node = None
            if request_id in session.tree_storage:
                node = session.tree_storage[request_id]
                if data.get("node_id") is not None:
                    node = node.get_node(data["node_id"])

            if node is None:
                node = TokenNode(
                    id="root",
                    token_id=-1,
                    text=data["input_text"],
                    prob=1.0,
                    score=1.0,
                    depth=0,
                    parent=None,
                    children={}
                )

            await websocket.send_json({
                "type": "generation_status",
                "request_id": request_id,
                "status": "started",
                "message": "Starting BFS tree generation..."
            })

            self.model_manager.inference_engine.set_broadcast_queue(session.broadcast_queue)

            engine = self.model_manager.inference_engine
            generation_kwargs = dict(
                node=node,
                k=data.get("k", 5),
                particlenum=data.get("particlenum", data.get("num_responses", 20)),
                max_tokens=data.get("max_tokens", 50),
                depth=data.get("depth"),
                temperature=data.get("temperature", 0.7),
                top_p=data.get("top_p", 0.9),
                min_p=data.get("min_p", 0.05)
            )
            if hasattr(engine, "generate_with_bfs"):
                tree = await engine.generate_with_bfs(**generation_kwargs)
            else:
                tree = await engine.generate_with_smc(**generation_kwargs)

            session.tree_storage[request_id] = tree
            await websocket.send_json({
                "request_id": request_id,
                "type": "tree_result",
                "tree": tree.to_dict()
            })

        except Exception as e:
            await self.model_manager.handle_inference_error(websocket, request_id, e)

    async def _handle_generate_with_smc(
        self,
        websocket: WebSocket,
        session: SessionState,
        data: Dict[str, Any]
    ) -> None:
        """Handle legacy SMC request name as BFS generation."""
        await self._handle_generate_with_bfs(websocket, session, data)
