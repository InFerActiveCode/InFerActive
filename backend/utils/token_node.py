"""Token node data structure for tree-based inference."""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, TYPE_CHECKING
from transformers import AutoTokenizer

if TYPE_CHECKING:
    from utils.token_node import TokenNode


@dataclass
class TokenNode:
    """
    Represents a token node in a generation tree.
    
    Each node contains a token and its associated probability/score,
    along with parent-child relationships forming a tree structure.
    """
    
    id: str
    token_id: int
    text: str
    prob: float
    score: float
    depth: int
    parent: Optional['TokenNode']
    children: Dict[str, 'TokenNode'] = field(default_factory=dict)
    
    def to_dict(self) -> dict:
        """
        Convert the node and its subtree to a dictionary representation.
        
        Returns:
            dict: Dictionary representation of the node tree.
        """
        return {
            "id": self.id,
            "token_id": self.token_id,
            "text": self.text,
            "prob": self.prob,
            "score": self.score,
            "children": [child.to_dict() for child in self.children.values()]
        }
    
    def get_token_prefix(self, tokenizer: AutoTokenizer, model_type: str = "llama") -> List[int]:
        """
        Get the complete token sequence from root to this node.
        
        For root nodes, applies the appropriate chat template based on model type.
        For non-root nodes, recursively builds the token sequence.
        
        Args:
            tokenizer: The tokenizer to use for encoding.
            model_type: Type of model (llama, qwen, exaone).
            
        Returns:
            List[int]: Token IDs representing the sequence from root to this node.
        """
        if self.id == "root":
            return self._build_root_prompt(tokenizer, model_type)
        elif self.parent is None:
            return [self.token_id]
        else:
            return self.parent.get_token_prefix(tokenizer, model_type) + [self.token_id]
    
    def _build_root_prompt(self, tokenizer: AutoTokenizer, model_type: str) -> List[int]:
        """
        Build the tokenized prompt for root node with appropriate chat template.
        
        Args:
            tokenizer: The tokenizer to use.
            model_type: Type of model.
            
        Returns:
            List[int]: Encoded token sequence with chat template.
        """
        if model_type == "qwen":
            messages = [{"role": "user", "content": self.text}]
            instruct_prompt = tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=True  # For Qwen models with thinking capability
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
        else:  # llama
            instruct_prompt = (
                "<|begin_of_text|>"
                "<|start_header_id|>system<|end_header_id|>\n"
                "Cutting Knowledge Date: December 2023\n"
                "Today Date: 27 March 2025\n"
                "You are a helpful assistant<|eot_id|>"
                "<|start_header_id|>user<|end_header_id|>\n"
                f"{self.text}<|eot_id|>"
                "<|start_header_id|>assistant<|end_header_id|> "
            )
        
        return tokenizer.encode(instruct_prompt)
    
    def get_node(self, node_id: str) -> Optional['TokenNode']:
        """
        Search for a node in the subtree by ID.
        
        Args:
            node_id: ID of the node to find.
            
        Returns:
            TokenNode or None: The found node, or None if not found.
        """
        if self.id == node_id:
            return self
        
        for child in self.children.values():
            node = child.get_node(node_id)
            if node is not None:
                return node
        
        return None
    
    def get_all_nodes(self) -> List['TokenNode']:
        """
        Get all nodes in the subtree (including this node).
        
        Returns:
            List[TokenNode]: All nodes in the subtree.
        """
        nodes = [self]
        for child in self.children.values():
            nodes.extend(child.get_all_nodes())
        return nodes
    
    def collect_leaf_nodes(self) -> List['TokenNode']:
        """
        Collect all leaf nodes (nodes without children) in the subtree.
        
        Returns:
            List[TokenNode]: All leaf nodes in the subtree.
        """
        if not self.children:
            return [self]
        
        leaves = []
        for child in self.children.values():
            leaves.extend(child.collect_leaf_nodes())
        return leaves
    
    def trace_path_to_root(self) -> List['TokenNode']:
        """
        Trace the path from this node back to the root.
        
        Returns:
            List[TokenNode]: Path from root to this node.
        """
        path = []
        current = self
        while current is not None:
            path.insert(0, current)
            current = current.parent
        return path
    
    def build_text_sequence(self) -> str:
        """
        Build the complete text sequence from root to this node.
        
        Returns:
            str: Complete text sequence.
        """
        path = self.trace_path_to_root()
        if path and path[0].id == "root":
            # Skip root node text, start from first actual token
            return "".join(node.text for node in path[1:])
        return "".join(node.text for node in path)
    
    def __repr__(self) -> str:
        """String representation of the node."""
        return (
            f"TokenNode(id='{self.id}', token_id={self.token_id}, "
            f"text='{self.text}', prob={self.prob:.4f}, depth={self.depth})"
        )