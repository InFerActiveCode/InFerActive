"""Token node data structure for tree-based inference."""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

try:
    from .model_utils import build_prompt
except ImportError:
    from model_utils import build_prompt


@dataclass
class TokenNode:
    """Token node in a generation tree."""

    id: str
    token_id: int
    text: str
    prob: float
    score: float
    depth: int
    parent: Optional["TokenNode"]
    children: Dict[int, "TokenNode"] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "token_id": self.token_id,
            "text": self.text,
            "prob": self.prob,
            "score": self.score,
            "children": [child.to_dict() for child in self.children.values()]
        }

    @classmethod
    def from_dict(cls, data: dict, parent: Optional["TokenNode"] = None, depth: int = 0) -> "TokenNode":
        node = cls(
            id=data.get("id", f"node_{depth}"),
            token_id=data.get("token_id", -1),
            text=data.get("text", ""),
            prob=data.get("prob", 1.0),
            score=data.get("score", 1.0),
            depth=depth,
            parent=parent,
            children={}
        )
        for child_data in data.get("children", []):
            child = cls.from_dict(child_data, parent=node, depth=depth + 1)
            node.children[child.token_id] = child
        return node

    def get_token_prefix(self, tokenizer: Any, model_type: str = "llama2") -> List[int]:
        if self.id == "root":
            return self._build_root_prompt(tokenizer, model_type)
        if self.parent is None:
            return [self.token_id]
        return self.parent.get_token_prefix(tokenizer, model_type) + [self.token_id]

    def _build_root_prompt(self, tokenizer: Any, model_type: str) -> List[int]:
        prompt = build_prompt(self.text, model_type, tokenizer)
        return tokenizer.encode(prompt, add_special_tokens=True)

    def get_node(self, node_id: str) -> Optional["TokenNode"]:
        if self.id == node_id:
            return self

        for child in self.children.values():
            node = child.get_node(node_id)
            if node is not None:
                return node

        return None

    def get_all_nodes(self) -> List["TokenNode"]:
        nodes = [self]
        for child in self.children.values():
            nodes.extend(child.get_all_nodes())
        return nodes

    def collect_leaf_nodes(self) -> List["TokenNode"]:
        if not self.children:
            return [self]

        leaves = []
        for child in self.children.values():
            leaves.extend(child.collect_leaf_nodes())
        return leaves

    def trace_path_to_root(self) -> List["TokenNode"]:
        path = []
        current = self
        while current is not None:
            path.insert(0, current)
            current = current.parent
        return path

    def build_text_sequence(self) -> str:
        path = self.trace_path_to_root()
        if path and path[0].id == "root":
            return "".join(node.text for node in path[1:])
        return "".join(node.text for node in path)

    def __repr__(self) -> str:
        return (
            f"TokenNode(id='{self.id}', token_id={self.token_id}, "
            f"text='{self.text}', prob={self.prob:.4f}, depth={self.depth})"
        )
