#!/usr/bin/env python3
"""
Unified tree generator that creates complete token trees and saves them as JSON files.
Updated to work with the current backend architecture.
"""

import os
import sys
import asyncio
import json
import hashlib
import torch
import re
from pathlib import Path
from datetime import datetime
from typing import Optional, List

# Add parent directory to path to import backend modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from transformers import AutoModelForCausalLM, AutoTokenizer

# Import classes directly to avoid circular imports
import importlib
import sys

# Set GPU device
os.environ["CUDA_VISIBLE_DEVICES"] = "0"


class TreeGenerator:
    """
    Standalone tree generator that creates complete token trees
    and saves them as JSON files for offline analysis
    """

    def __init__(self, model_path: str, output_dir: str = "generated_trees"):
        """
        Initialize the tree generator

        Args:
            model_path: Path to the Llama model
            output_dir: Directory to save generated tree files
        """
        self.model_path = model_path
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)

        self.model = None
        self.tokenizer = None
        self.inference = None

    async def initialize(self):
        """Load model and tokenizer, initialize TokenInferenceEngine"""
        print(f"Loading tokenizer from {self.model_path}...")
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_path)

        print(f"Loading model...")
        # Load model exactly like backend's main.py
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_path,
            torch_dtype=torch.float16
        )
        self.model.eval()

        # Import TokenNode first to avoid circular imports
        print("Initializing inference engine...")

        # Import TokenNode directly before any utils imports
        from utils.token_node import TokenNode
        self.TokenNode = TokenNode

        # Now import TokenInferenceEngine
        from inference.engine import TokenInferenceEngine

        self.inference = TokenInferenceEngine(
            model=self.model,
            tokenizer=self.tokenizer,
            model_type="llama"
        )

        # Create broadcast queue after initialization
        self.inference.set_broadcast_queue(asyncio.Queue())

        print("Initialization complete!")

    async def generate_tree(
        self,
        prompt: str,
        exploration_depths: List[int] = None,
        k: int = 5,
        temperature: float = 0.7,
        top_p: float = 0.9,
        min_p: float = 0.05,
        output_filename: Optional[str] = None
    ) -> str:
        """
        Generate a complete tree and save it to a JSON file

        Args:
            prompt: Starting text for generation
            exploration_depths: Depth levels for recursive exploration (default: [3, 2])
            k: Top-k sampling parameter
            temperature: Temperature parameter for sampling
            top_p: Nucleus sampling threshold
            min_p: Minimum probability threshold
            output_filename: Custom output filename (auto-generated if None)

        Returns:
            Path to the saved JSON file
        """
        if exploration_depths is None:
            exploration_depths = [3, 2]

        print(f"\nGenerating tree for prompt: '{prompt[:50]}...'")
        print(f"Parameters: depths={exploration_depths}, k={k}, temp={temperature}, top_p={top_p}, min_p={min_p}")

        # Create root node
        root = self.TokenNode(
            id="root",
            token_id=-1,
            text=prompt,
            prob=1.0,
            score=1.0,
            depth=0,
            parent=None,
            children={}
        )

        # Use backend's exact generation approach - single explore_node call
        print(f"\nExploring tree with depths: {exploration_depths}")

        # Backend just calls explore_node once on root for initial generation
        await self.inference.explore_node(
            node=root,
            depth_to_explore=exploration_depths[0],  # Use first depth
            k=k,
            temperature=temperature,
            top_p=top_p,
            min_p=min_p,
            extend_greedy=False  # Disable greedy path extension
        )

        # If there are additional exploration phases, do them from leaf nodes
        if len(exploration_depths) > 1:
            print("\nPerforming additional exploration phases...")

            for phase_idx, depth in enumerate(exploration_depths[1:], 1):
                print(f"Phase {phase_idx + 1}: Exploring depth {depth}")

                # Get all leaf nodes
                leaf_nodes = self._get_leaf_nodes(root)
                # Filter out EOS tokens
                leaf_nodes = [n for n in leaf_nodes if n.token_id != self.tokenizer.eos_token_id]

                print(f"  Found {len(leaf_nodes)} non-EOS leaf nodes")

                # Explore from each leaf
                for node in leaf_nodes:
                    await self.inference.explore_node(
                        node=node,
                        depth_to_explore=depth,
                        k=k,
                        temperature=temperature,
                        top_p=top_p,
                        min_p=min_p,
                        extend_greedy=False  # Disable greedy path extension
                    )


        # Convert tree to dictionary
        print("\nConverting tree to dictionary format...")
        tree_dict = root.to_dict()

        # Calculate statistics after pruning
        total_nodes = self._count_nodes(root)
        max_depth = self._get_max_depth(root)

        # Get and display leaf nodes
        leaf_nodes = self._get_leaf_nodes(root)
        eos_nodes = [n for n in leaf_nodes if n.token_id == self.tokenizer.eos_token_id]
        other_nodes = [n for n in leaf_nodes if n.token_id != self.tokenizer.eos_token_id]

        # Calculate average depth of leaf nodes
        avg_depth = sum(n.depth for n in leaf_nodes) / len(leaf_nodes) if leaf_nodes else 0

        print(f"\n=== Leaf Nodes Summary ===")
        print(f"Total leaf nodes: {len(leaf_nodes)}")
        print(f"  - EOS token nodes: {len(eos_nodes)}")
        print(f"  - Other nodes: {len(other_nodes)}")

        # Create metadata
        metadata = {
            "prompt": prompt,
            "model_path": self.model_path,
            "parameters": {
                "exploration_depths": exploration_depths,
                "k": k,
                "temperature": temperature,
                "top_p": top_p,
                "min_p": min_p
            },
            "statistics": {
                "total_nodes": total_nodes,
                "leaf_nodes": len(leaf_nodes),
                "eos_token_nodes": len(eos_nodes),
                "other_nodes": len(other_nodes),
                "max_depth": max_depth,
                "avg_depth": round(avg_depth, 2)
            },
            "generated_at": datetime.now().isoformat()
        }

        # Combine metadata and tree
        output_data = {
            "metadata": metadata,
            "tree": tree_dict
        }

        # Generate filename if not provided
        if output_filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

            # Extract user prompt prefix (first 10 chars, cleaned)
            # Try to find "User:" pattern in the prompt
            if "User:" in prompt:
                user_part = prompt.split("User:", 1)[1].strip()
            else:
                user_part = prompt

            # Clean the prefix: remove special chars, keep alphanumeric and spaces
            prefix_clean = re.sub(r'[^a-zA-Z0-9\s]', '', user_part[:10])
            prefix_clean = prefix_clean.replace(' ', '_').lower()[:10]

            # Ensure we have something, fallback to hash if empty
            if not prefix_clean:
                prompt_hash = hashlib.md5(prompt.encode()).hexdigest()[:8]
                output_filename = f"tree_{timestamp}_{prompt_hash}.json"
            else:
                output_filename = f"{prefix_clean}_{timestamp}.json"

        output_path = self.output_dir / output_filename

        # Save to file
        print(f"\nSaving tree to {output_path}...")
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(output_data, f, ensure_ascii=False, indent=2)

        print(f"\n✓ Tree saved successfully!")
        print(f"  Total nodes: {total_nodes}")
        print(f"  Leaf nodes: {len(leaf_nodes)}")
        print(f"    - EOS: {len(eos_nodes)}")
        print(f"    - Other: {len(other_nodes)}")
        print(f"  Depth: Max={max_depth}, Avg={avg_depth:.1f}")
        print(f"  File size: {output_path.stat().st_size / 1024:.1f} KB")

        return str(output_path), root

    def _get_leaf_nodes(self, node) -> List:
        """Recursively collect all leaf nodes"""
        if not node.children:
            return [node]
        leaves = []
        for child in node.children.values():
            leaves.extend(self._get_leaf_nodes(child))
        return leaves

    def _count_nodes(self, node) -> int:
        """Count total number of nodes in the tree"""
        count = 1
        for child in node.children.values():
            count += self._count_nodes(child)
        return count

    def _get_max_depth(self, node) -> int:
        """Get maximum depth of the tree"""
        if not node.children:
            return node.depth
        return max(self._get_max_depth(child) for child in node.children.values())

async def main():
    """Main execution function"""

    # Configuration
    MODEL_PATH = "../models/llama-3.1-8b-instruct_fp16"
    OUTPUT_DIR = "../generated_trees"

    # ========== PROMPT CONFIGURATION ==========
    prompt = "How many r in strawberry?"
    # ==========================================

    # Create generator
    generator = TreeGenerator(MODEL_PATH, OUTPUT_DIR)
    await generator.initialize()

    # Generate tree
    print("\n" + "="*60)
    print("Generating tree")
    print("="*60)

    filepath, root = await generator.generate_tree(
        prompt=prompt,
        exploration_depths=[3, 3, 3, 3, 3,
                            3, 3, 3, 3, 3,
                            3, 3, 3, 3, 3,
                            3, 3, 3, 3, 3],
        k=5,
        top_p=0.9,
        temperature=0.7
    )

    print("\n" + "="*60)
    print(f"Tree generated successfully! Saved to: {filepath}")
    print("="*60)

    # Clean up GPU memory
    torch.cuda.empty_cache()


if __name__ == "__main__":
    asyncio.run(main())