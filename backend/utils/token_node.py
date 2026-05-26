"""Token node data structure for tree-based inference."""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class TokenNode:
    """Token node used by the interactive generation tree."""

    id: str
    token_id: int
    text: str
    prob: float
    score: float
    depth: int
    parent: Optional["TokenNode"]
    children: Dict[str, "TokenNode"] = field(default_factory=dict)
    
    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "token_id": self.token_id,
            "text": self.text,
            "prob": self.prob,
            "score": self.score,
            "children": [child.to_dict() for child in self.children.values()]
        }

    def get_token_prefix(self, tokenizer: Any, model_type: str = "llama") -> List[int]:
        if self.id == "root":
            return self._build_root_prompt(tokenizer, model_type)
        if self.parent is None:
            return [self.token_id]
        return self.parent.get_token_prefix(tokenizer, model_type) + [self.token_id]

    def _build_root_prompt(self, tokenizer: Any, model_type: str) -> List[int]:
        if model_type == "qwen":
            messages = [{"role": "user", "content": self.text}]
            instruct_prompt = tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=True
            )
        elif model_type == "exaone":
            messages = [
                {
                    "role": "system",
                    "content": "You are EXAONE model from LG AI Research, a helpful assistant."
                },
                {"role": "user", "content": self.text}
            ]
            instruct_prompt = tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True
            )
        else:
            instruct_prompt = (
                "<|begin_of_text|>"
                "<|start_header_id|>system<|end_header_id|>\n"
                "You are a helpful assistant<|eot_id|>"
                "<|start_header_id|>user<|end_header_id|>\n"
                f"{self.text}<|eot_id|>"
                "<|start_header_id|>assistant<|end_header_id|> "
            )

        return tokenizer.encode(instruct_prompt)

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
