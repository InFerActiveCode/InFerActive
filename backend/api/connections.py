"""WebSocket connection and session management."""

import asyncio
import logging
import uuid
from typing import Dict, Optional, TYPE_CHECKING

from fastapi import WebSocket

if TYPE_CHECKING:
    from utils.token_node import TokenNode

logger = logging.getLogger(__name__)


class WebSocketConnectionManager:
    """Manages WebSocket connections and broadcasting."""
    
    def __init__(self):
        self.active_connections: Dict[str, WebSocket] = {}
        
    async def connect(self, websocket: WebSocket) -> str:
        """
        Accept a new WebSocket connection and assign it an ID.
        
        Args:
            websocket: The WebSocket connection to accept.
            
        Returns:
            str: Unique connection ID.
        """
        await websocket.accept()
        connection_id = str(uuid.uuid4())
        self.active_connections[connection_id] = websocket
        
        logger.info(f"WebSocket connected: {connection_id}")
        return connection_id
    
    def disconnect(self, connection_id: str) -> None:
        """
        Remove a connection from active connections.
        
        Args:
            connection_id: ID of the connection to remove.
        """
        if connection_id in self.active_connections:
            del self.active_connections[connection_id]
            logger.info(f"WebSocket disconnected: {connection_id}")
    
    async def send_to_connection(self, connection_id: str, message: dict) -> bool:
        """
        Send a message to a specific connection.
        
        Args:
            connection_id: ID of the target connection.
            message: Message to send.
            
        Returns:
            bool: True if message was sent successfully, False otherwise.
        """
        if connection_id not in self.active_connections:
            return False
        
        try:
            await self.active_connections[connection_id].send_json(message)
            return True
        except Exception as e:
            logger.error(f"Failed to send message to {connection_id}: {e}")
            return False
    
    async def broadcast_to_all(self, message: dict) -> int:
        """
        Broadcast a message to all active connections.
        
        Args:
            message: Message to broadcast.
            
        Returns:
            int: Number of connections the message was sent to.
        """
        sent_count = 0
        failed_connections = []
        
        for connection_id, websocket in self.active_connections.items():
            try:
                await websocket.send_json(message)
                sent_count += 1
            except Exception as e:
                logger.warning(f"Failed to broadcast to {connection_id}: {e}")
                failed_connections.append(connection_id)
        
        # Clean up failed connections
        for conn_id in failed_connections:
            self.disconnect(conn_id)
        
        return sent_count
    
    def get_connection_count(self) -> int:
        """Get the number of active connections."""
        return len(self.active_connections)
    
    def get_connection_ids(self) -> list[str]:
        """Get all active connection IDs."""
        return list(self.active_connections.keys())


class SessionState:
    """Manages state for a WebSocket session."""
    
    def __init__(self, connection_id: str):
        self.connection_id = connection_id
        self.tree_storage: Dict[str, 'TokenNode'] = {}
        self.broadcast_task: Optional[asyncio.Task] = None
        self.broadcast_queue: asyncio.Queue = asyncio.Queue()
        
        logger.info(f"Session created: {connection_id}")
    
    async def start_broadcast_worker(self, connection_manager: WebSocketConnectionManager) -> None:
        """Start the background broadcast task."""
        if self.broadcast_task is None:
            self.broadcast_task = asyncio.create_task(
                self._run_broadcast_worker(connection_manager)
            )
    
    async def stop_broadcast_worker(self) -> None:
        """Stop the background broadcast task."""
        if self.broadcast_task:
            self.broadcast_task.cancel()
            try:
                await self.broadcast_task
            except asyncio.CancelledError:
                pass
            finally:
                self.broadcast_task = None
    
    async def _run_broadcast_worker(self, connection_manager: WebSocketConnectionManager) -> None:
        """Background worker to handle broadcast messages for this session."""
        while True:
            try:
                # Wait for messages with a timeout to allow for cancellation
                try:
                    message = await asyncio.wait_for(
                        self.broadcast_queue.get(), timeout=0.1
                    )
                    await connection_manager.send_to_connection(
                        self.connection_id, message
                    )
                except asyncio.TimeoutError:
                    continue
                    
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Broadcast error in session {self.connection_id}: {e}")
                await asyncio.sleep(0.5)
    
    def __del__(self):
        """Cleanup when session is garbage collected."""
        if self.broadcast_task and not self.broadcast_task.done():
            logger.warning(f"Session {self.connection_id} deleted with active broadcast task")


class SessionManager:
    """Manages active WebSocket sessions."""
    
    def __init__(self):
        self.active_sessions: Dict[str, SessionState] = {}
        
    async def create_session(
        self, 
        connection_id: str, 
        connection_manager: WebSocketConnectionManager
    ) -> SessionState:
        """
        Create a new session for a connection.
        
        Args:
            connection_id: ID of the connection.
            connection_manager: Connection manager instance.
            
        Returns:
            SessionState: The created session.
        """
        session = SessionState(connection_id)
        self.active_sessions[connection_id] = session
        await session.start_broadcast_worker(connection_manager)
        
        logger.info(f"Session started: {connection_id}")
        return session
    
    async def remove_session(self, connection_id: str) -> None:
        """
        Remove a session.
        
        Args:
            connection_id: ID of the connection to remove.
        """
        if connection_id in self.active_sessions:
            session = self.active_sessions[connection_id]
            await session.stop_broadcast_worker()
            del self.active_sessions[connection_id]
            
            logger.info(f"Session removed: {connection_id}")
    
    def get_session(self, connection_id: str) -> Optional[SessionState]:
        """
        Get a session by connection ID.
        
        Args:
            connection_id: ID of the connection.
            
        Returns:
            SessionState or None: The session if it exists.
        """
        return self.active_sessions.get(connection_id)
    
    async def cleanup_all_sessions(self) -> None:
        """Clean up all active sessions."""
        session_ids = list(self.active_sessions.keys())
        for session_id in session_ids:
            await self.remove_session(session_id)
        
        logger.info("All sessions cleaned up")
    
    def get_session_count(self) -> int:
        """Get the number of active sessions."""
        return len(self.active_sessions)
    
    def get_session_ids(self) -> list[str]:
        """Get all active session IDs."""
        return list(self.active_sessions.keys())