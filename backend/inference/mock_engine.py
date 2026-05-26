"""Mock inference engine backed by a precomputed token tree JSON file."""

import asyncio
import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

from utils.token_node import TokenNode

logger = logging.getLogger(__name__)


class MockTreeInferenceEngine:
    """Serve deterministic tree updates from a saved evaluation tree.

    This engine is intended for frontend/backend protocol testing on machines
    that cannot run vLLM. It preserves the same async methods used by the real
    inference engine.
    """

    is_mock = True

    def __init__(
        self,
        tree_path: Path,
        initial_depth: int = 5,
        expand_depth: int = 4,
        postfix: str = " this is mockup sending",
        delay_seconds: float = 0.05,
    ):
        self.tree_path = Path(tree_path)
        self.initial_depth = max(0, initial_depth)
        self.expand_depth = max(0, expand_depth)
        self.postfix = postfix
        self.delay_seconds = max(0.0, delay_seconds)
        self._broadcast_queue: Optional[asyncio.Queue] = None

        payload = json.loads(self.tree_path.read_text(encoding="utf-8"))
        tree_data = payload.get("tree", payload)
        self.metadata = payload.get("metadata", {})
        self.source_root = self._node_from_dict(tree_data, depth=0, parent=None)

        logger.info(
            "MockTreeInferenceEngine loaded %s (%s nodes)",
            self.tree_path,
            len(self.source_root.get_all_nodes()),
        )

    def set_broadcast_queue(self, queue: asyncio.Queue) -> None:
        """Set the broadcast queue for real-time updates."""
        self._broadcast_queue = queue

    async def generate_with_bfs(
        self,
        node: TokenNode,
        k: int = 5,
        particlenum: int = 20,
        max_tokens: int = 15,
        temperature: float = 0.7,
        top_p: float = 0.9,
        min_p: float = 0.05,
        depth: Optional[int] = None,
    ) -> TokenNode:
        """Return a depth-limited copy of the mock tree."""
        await self._sleep()
        limit = self._resolve_depth(depth, self.initial_depth)
        root = self._clone_limited(self.source_root, limit)
        root.text = node.text
        return root

    async def generate_with_smc(self, *args: Any, **kwargs: Any) -> TokenNode:
        """Compatibility alias for older frontend/backend clients."""
        return await self.generate_with_bfs(*args, **kwargs)

    async def explore_node(
        self,
        node: TokenNode,
        depth_to_explore: int,
        k: int = 5,
        temperature: float = 0.7,
        top_p: float = 0.9,
        min_p: float = 0.05,
        extend_greedy: bool = True,
    ) -> None:
        """Expand a node from the saved tree and broadcast an update."""
        await self._sleep()
        limit = min(self._resolve_depth(depth_to_explore, self.expand_depth), self.expand_depth)
        source_node = self.source_root.get_node(node.id)

        if source_node is None:
            logger.warning("Mock source node not found for %s; returning postfix only", node.id)
            subtree = TokenNode(
                id=node.id,
                token_id=node.token_id,
                text=node.text,
                prob=node.prob,
                score=node.score,
                depth=node.depth,
                parent=None,
                children={},
            )
            self._attach_postfix(subtree)
        else:
            subtree = self._clone_limited(source_node, limit)

        node.token_id = subtree.token_id
        node.text = subtree.text
        node.prob = subtree.prob
        node.score = subtree.score
        node.children = subtree.children
        for child in node.children.values():
            child.parent = node

        if self._broadcast_queue:
            await self._broadcast_queue.put({
                "type": "update",
                "tree": node.to_dict(),
            })

    def _resolve_depth(self, requested: Optional[int], fallback: int) -> int:
        try:
            if requested is None:
                return fallback
            return max(0, int(requested))
        except (TypeError, ValueError):
            return fallback

    async def _sleep(self) -> None:
        if self.delay_seconds > 0:
            await asyncio.sleep(self.delay_seconds)

    def _clone_limited(self, source: TokenNode, remaining_depth: int) -> TokenNode:
        clone = TokenNode(
            id=source.id,
            token_id=source.token_id,
            text=source.text,
            prob=source.prob,
            score=source.score,
            depth=source.depth,
            parent=None,
            children={},
        )

        if remaining_depth <= 0 or not source.children:
            self._attach_postfix(clone)
            return clone

        for key, child in source.children.items():
            child_clone = self._clone_limited(child, remaining_depth - 1)
            child_clone.parent = clone
            clone.children[key] = child_clone

        return clone

    def _attach_postfix(self, node: TokenNode) -> None:
        if not self.postfix or node.children:
            return

        current = node
        for index, token_text in enumerate(self._split_postfix(self.postfix)):
            child = TokenNode(
                id=f"{current.id}:mock:{index}",
                token_id=-(index + 2),
                text=token_text,
                prob=1.0,
                score=1.0,
                depth=current.depth + 1,
                parent=current,
                children={},
            )
            current.children[child.id] = child
            current = child

    @staticmethod
    def _split_postfix(postfix: str) -> List[str]:
        words = postfix.strip().split()
        if not words:
            return []
        return [f" {word}" for word in words]

    def _node_from_dict(
        self,
        data: Dict[str, Any],
        depth: int,
        parent: Optional[TokenNode],
    ) -> TokenNode:
        node = TokenNode(
            id=str(data.get("id", f"mock-{depth}")),
            token_id=int(data.get("token_id", -1)),
            text=str(data.get("text", "")),
            prob=float(data.get("prob", 1.0)),
            score=float(data.get("score", data.get("prob", 1.0))),
            depth=int(data.get("depth", depth)),
            parent=parent,
            children={},
        )

        for index, child_data in enumerate(data.get("children", [])):
            child = self._node_from_dict(child_data, depth + 1, node)
            node.children[f"{child.id}:{index}"] = child

        return node
