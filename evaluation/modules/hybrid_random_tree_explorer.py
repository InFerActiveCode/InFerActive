"""Hybrid random-walk tree resumer backed by vLLM processed logprobs.

This module intentionally lives beside the existing explorers instead of
modifying them. It loads precomputed DFS trees, optionally resumes expansion,
then runs uniform random walks over the materialized tree. When a walk reaches
a non-EOS leaf before the target depth, only that leaf is expanded on demand.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import math
import os
import random
import shutil
import sys
import tempfile
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import torch

if __name__ == "__main__":
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

try:
    from .token_node import TokenNode
    from .model_utils import set_system_prompt_suffix
    from .tree_explorer_vllm import (
        MAX_DEPTH,
        TreeExplorerVLLM,
        get_tree_file_candidates,
        load_behaviors,
        validate_tree_json,
    )
except ImportError:
    from token_node import TokenNode
    from model_utils import set_system_prompt_suffix
    from tree_explorer_vllm import (
        MAX_DEPTH,
        TreeExplorerVLLM,
        get_tree_file_candidates,
        load_behaviors,
        validate_tree_json,
    )


HYBRID_TREE_BACKEND = "vllm_processed_logprobs_hybrid_uniform_walk"
DEFAULT_TARGET_DEPTH = 20
DEFAULT_NUM_WALKS = 1000
DEFAULT_PREEXPAND_MAX_NODES = 10000


def _stable_seed(*parts: Any) -> int:
    seed_text = ":".join(str(part) for part in parts)
    digest = hashlib.sha256(seed_text.encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big", signed=False)


def _node_extra_dict(node: TokenNode, key: str) -> Dict[str, int]:
    value = getattr(node, key, None)
    if isinstance(value, dict):
        return {str(k): int(v) for k, v in value.items() if int(v) != 0}
    return {}


class HybridRandomWalkTreeExplorerVLLM(TreeExplorerVLLM):
    """Resume existing trees and run seed-indexed uniform random walks."""

    def __init__(
        self,
        *args: Any,
        input_tree_dir: str,
        behaviors_csv: Optional[str] = None,
        categories: Optional[List[str]] = None,
        target_depth: int = DEFAULT_TARGET_DEPTH,
        num_walks: int = DEFAULT_NUM_WALKS,
        num_samples: int = 1,
        seed: int = 42,
        walk_batch_size: int = DEFAULT_NUM_WALKS,
        preexpand_min_depth: Optional[int] = None,
        preexpand_max_nodes: int = DEFAULT_PREEXPAND_MAX_NODES,
        **kwargs: Any,
    ):
        super().__init__(*args, **kwargs)
        self.input_tree_dir = Path(input_tree_dir)
        self.behaviors_csv = behaviors_csv
        self.categories = categories
        self.target_depth = int(target_depth)
        self.num_walks = int(num_walks)
        self.num_samples = int(num_samples)
        self.seed = int(seed)
        self.walk_batch_size = int(walk_batch_size)
        self.preexpand_max_nodes = int(preexpand_max_nodes)

    def _validate_parameters(self) -> None:
        if self.target_depth < 1:
            raise ValueError("target_depth must be >= 1")
        if self.num_walks < 1:
            raise ValueError("num_walks must be >= 1")
        if self.num_samples < 1:
            raise ValueError("num_samples must be >= 1")
        if self.walk_batch_size < 1:
            raise ValueError("walk_batch_size must be >= 1")
        if self.batch_size < 1:
            raise ValueError("max batch size must be >= 1")
        if self.preexpand_max_nodes < 0:
            raise ValueError("preexpand_max_nodes must be >= 0")
        if self.max_logprobs == -1:
            raise ValueError(
                "hybrid random walks require a finite --max-logprobs value "
                "to keep demand expansion bounded"
            )
        if self.max_logprobs < 1:
            raise ValueError("max_logprobs must be >= 1")

    def _is_eos(self, node: TokenNode, terminator_token_ids: set[int]) -> bool:
        return int(node.token_id) in terminator_token_ids

    def _is_terminal_for_walk(
        self,
        node: TokenNode,
        terminator_token_ids: set[int],
    ) -> bool:
        return (
            self._is_eos(node, terminator_token_ids)
            or int(node.depth) >= self.target_depth
            or bool(getattr(node, "_terminal_reason", ""))
        )

    def _terminal_reason(
        self,
        node: TokenNode,
        terminator_token_ids: set[int],
    ) -> str:
        existing = str(getattr(node, "_terminal_reason", ""))
        if existing:
            return existing
        if self._is_eos(node, terminator_token_ids):
            return "eos"
        if int(node.depth) >= self.target_depth:
            return "max_depth"
        return "dead_end"

    def _mark_terminal(self, node: TokenNode, reason: str) -> None:
        setattr(node, "_terminal_reason", reason)

    def _increment_seed_count(self, node: TokenNode, attr: str, seed_label: str) -> None:
        counts = getattr(node, attr, None)
        if not isinstance(counts, dict):
            counts = {}
        counts[seed_label] = int(counts.get(seed_label, 0)) + 1
        setattr(node, attr, counts)

    def _mark_walk_hit(
        self,
        node: TokenNode,
        seed_label: str,
        terminator_token_ids: set[int],
    ) -> str:
        reason = self._terminal_reason(node, terminator_token_ids)
        self._mark_terminal(node, reason)
        self._increment_seed_count(node, "_walk_counts", seed_label)
        return reason

    def _subtree_node_count(self, node: TokenNode) -> int:
        return self._count_nodes(node)

    def _count_leaf_nodes(self, root: TokenNode) -> int:
        return len(self._get_leaf_nodes(root))

    def _prune_last_depth(self, root: TokenNode) -> int:
        """Remove every node at the current maximum depth."""
        max_depth = self._get_max_depth(root)
        if max_depth <= 0:
            return 0

        removed = 0

        def prune(node: TokenNode) -> None:
            nonlocal removed
            for token_id, child in list(node.children.items()):
                if child.depth == max_depth:
                    removed += self._subtree_node_count(child)
                    del node.children[token_id]
                else:
                    prune(child)

        prune(root)
        return removed

    def _nonterminal_leaves(
        self,
        root: TokenNode,
        terminator_token_ids: set[int],
    ) -> List[TokenNode]:
        return [
            node
            for node in self._get_leaf_nodes(root)
            if not self._is_eos(node, terminator_token_ids)
            and int(node.depth) < self.target_depth
            and not bool(getattr(node, "_terminal_reason", ""))
        ]

    def _expand_nodes(
        self,
        nodes: Sequence[TokenNode],
        k: int,
        temperature: float,
        top_p: float,
        min_p: float,
    ) -> Tuple[int, int, Counter[str]]:
        """Expand unique nodes by one token and return expansion stats."""
        assert self.llm is not None
        assert self.tokenizer is not None

        unique_nodes = list({node.id: node for node in nodes}.values())
        if not unique_nodes:
            return 0, 0, Counter()

        prompts = [
            {"prompt_token_ids": node.get_token_prefix(self.tokenizer, self.model_type)}
            for node in unique_nodes
        ]
        sampling_params = self._build_sampling_params(k, temperature, top_p, min_p)
        outputs = self.llm.generate(prompts, sampling_params=sampling_params)
        if len(outputs) != len(unique_nodes):
            raise RuntimeError(
                f"vLLM returned {len(outputs)} outputs for {len(unique_nodes)} nodes"
            )

        expanded_nodes = 0
        children_added = 0
        terminal_reasons: Counter[str] = Counter()

        for node, output in zip(unique_nodes, outputs):
            if not output.outputs:
                self._mark_terminal(node, "missing_output")
                terminal_reasons["missing_output"] += 1
                continue

            candidates = self._extract_candidates(output.outputs[0].logprobs)
            if not candidates:
                self._mark_terminal(node, "missing_logprobs")
                terminal_reasons["missing_logprobs"] += 1
                continue

            finite_probs = [
                math.exp(logprob)
                for _token_id, logprob, _rank, _decoded_token in candidates
                if math.isfinite(float(logprob))
            ]
            if not finite_probs or sum(finite_probs) <= 0.0:
                self._mark_terminal(node, "zero_candidate_mass")
                terminal_reasons["zero_candidate_mass"] += 1
                continue

            max_prob = max(finite_probs)
            for token_id, logprob, _rank, decoded_token in candidates:
                prob_value = float(math.exp(logprob))
                token_text = (
                    decoded_token
                    if decoded_token is not None
                    else self._decode_token(int(token_id))
                )
                existing_child = node.children.get(int(token_id))
                if existing_child is not None:
                    existing_child.text = token_text
                    existing_child.prob = prob_value
                    existing_child.score = prob_value
                    setattr(existing_child, "_is_greedy", prob_value >= max_prob)
                    continue

                child_node = TokenNode(
                    id=os.urandom(4).hex(),
                    token_id=int(token_id),
                    text=token_text,
                    prob=prob_value,
                    score=prob_value,
                    depth=node.depth + 1,
                    parent=node,
                    children={},
                )
                setattr(child_node, "_is_greedy", prob_value >= max_prob)
                node.children[int(token_id)] = child_node
                children_added += 1

            setattr(node, "_candidate_complete", True)
            expanded_nodes += 1

        return expanded_nodes, children_added, terminal_reasons

    def _expand_until_leaf_soft_target(
        self,
        root: TokenNode,
        k: int,
        temperature: float,
        top_p: float,
        min_p: float,
        terminator_token_ids: set[int],
    ) -> Dict[str, Any]:
        """Preexpand until leaf count reaches the target; overshoot is kept."""
        stats = {
            "expanded_nodes": 0,
            "children_added": 0,
            "generate_calls": 0,
            "batch_size_counts": Counter(),
            "terminal_reasons": Counter(),
            "start_total_nodes": self._count_nodes(root),
            "start_leaf_nodes": self._count_leaf_nodes(root),
            "end_total_nodes": None,
            "end_leaf_nodes": None,
        }

        if self.preexpand_max_nodes <= 0:
            stats["end_total_nodes"] = stats["start_total_nodes"]
            stats["end_leaf_nodes"] = stats["start_leaf_nodes"]
            return stats

        while self._count_leaf_nodes(root) < self.preexpand_max_nodes:
            frontier = self._nonterminal_leaves(root, terminator_token_ids)
            if not frontier:
                break

            for start in range(0, len(frontier), self.batch_size):
                if self._count_leaf_nodes(root) >= self.preexpand_max_nodes:
                    break
                batch = frontier[start:start + self.batch_size]
                if not batch:
                    continue
                expanded, added, terminal_reasons = self._expand_nodes(
                    batch,
                    k=k,
                    temperature=temperature,
                    top_p=top_p,
                    min_p=min_p,
                )
                stats["generate_calls"] += 1
                stats["batch_size_counts"][len(batch)] += 1
                stats["expanded_nodes"] += expanded
                stats["children_added"] += added
                stats["terminal_reasons"].update(terminal_reasons)

        stats["end_total_nodes"] = self._count_nodes(root)
        stats["end_leaf_nodes"] = self._count_leaf_nodes(root)
        return stats

    def _choose_uniform_child(
        self,
        node: TokenNode,
        rng: random.Random,
    ) -> Optional[TokenNode]:
        children = list(node.children.values())
        if not children:
            return None
        return rng.choice(children)

    def _run_walks_for_seed(
        self,
        root: TokenNode,
        seed_value: int,
        seed_label: str,
        k: int,
        temperature: float,
        top_p: float,
        min_p: float,
        terminator_token_ids: set[int],
    ) -> Dict[str, Any]:
        rng = random.Random(seed_value)
        completed_walks = 0
        unique_leaf_ids: set[str] = set()
        demand_expanded_nodes = 0
        demand_children_added = 0
        terminal_reason_counts: Counter[str] = Counter()
        demand_terminal_reasons: Counter[str] = Counter()

        while completed_walks < self.num_walks:
            batch_count = min(self.walk_batch_size, self.num_walks - completed_walks)
            active: List[TokenNode] = [root for _ in range(batch_count)]

            while active:
                nodes_to_expand: Dict[str, TokenNode] = {}
                for node in active:
                    if self._is_terminal_for_walk(node, terminator_token_ids):
                        continue
                    if not node.children:
                        nodes_to_expand[node.id] = node

                if nodes_to_expand:
                    nodes = list(nodes_to_expand.values())
                    for start in range(0, len(nodes), self.batch_size):
                        expanded, added, terminal_reasons = self._expand_nodes(
                            nodes[start:start + self.batch_size],
                            k=k,
                            temperature=temperature,
                            top_p=top_p,
                            min_p=min_p,
                        )
                        demand_expanded_nodes += expanded
                        demand_children_added += added
                        demand_terminal_reasons.update(terminal_reasons)

                next_active: List[TokenNode] = []
                for node in active:
                    if self._is_terminal_for_walk(node, terminator_token_ids):
                        reason = self._mark_walk_hit(
                            node,
                            seed_label=seed_label,
                            terminator_token_ids=terminator_token_ids,
                        )
                        terminal_reason_counts[reason] += 1
                        unique_leaf_ids.add(node.id)
                        completed_walks += 1
                        continue

                    child = self._choose_uniform_child(node, rng)
                    if child is None:
                        self._mark_terminal(node, "dead_end")
                        reason = self._mark_walk_hit(
                            node,
                            seed_label=seed_label,
                            terminator_token_ids=terminator_token_ids,
                        )
                        terminal_reason_counts[reason] += 1
                        unique_leaf_ids.add(node.id)
                        completed_walks += 1
                    else:
                        next_active.append(child)

                active = next_active

        return {
            "completed_walks": completed_walks,
            "unique_leaf_hits": len(unique_leaf_ids),
            "duplicate_leaf_hits": max(0, completed_walks - len(unique_leaf_ids)),
            "demand_expanded_nodes": demand_expanded_nodes,
            "demand_children_added": demand_children_added,
            "terminal_reasons": dict(sorted(terminal_reason_counts.items())),
            "demand_terminal_reasons": dict(sorted(demand_terminal_reasons.items())),
        }

    def _prune_unwalked_paths(self, root: TokenNode) -> None:
        """Keep only paths that end at a walked node."""

        def prune(node: TokenNode) -> bool:
            kept_children: Dict[int, TokenNode] = {}
            for token_id, child in node.children.items():
                if prune(child):
                    kept_children[token_id] = child
            node.children = kept_children
            walk_counts = _node_extra_dict(node, "_walk_counts")
            return node.id == "root" or bool(node.children) or bool(walk_counts)

        prune(root)

    def _node_to_dict(self, node: TokenNode) -> Dict[str, Any]:
        data: Dict[str, Any] = {
            "id": node.id,
            "token_id": node.token_id,
            "text": node.text,
            "prob": node.prob,
            "score": node.score,
            "children": [self._node_to_dict(child) for child in node.children.values()],
        }

        walk_counts = _node_extra_dict(node, "_walk_counts")
        if walk_counts:
            data["walk_counts"] = walk_counts
            data["walk_count"] = sum(walk_counts.values())

        terminal_reason = str(getattr(node, "_terminal_reason", ""))
        if terminal_reason:
            data["terminal_reason"] = terminal_reason

        if hasattr(node, "_is_greedy"):
            data["is_greedy"] = bool(getattr(node, "_is_greedy"))
        if hasattr(node, "_candidate_complete"):
            data["candidate_complete"] = bool(getattr(node, "_candidate_complete"))

        return data

    def _load_tree_file(self, path: Path) -> Tuple[Dict[str, Any], TokenNode]:
        with open(path, "r", encoding="utf-8") as handle:
            data = json.load(handle)
        root = TokenNode.from_dict(data["tree"])
        return data, root

    def _all_leaves_eos(
        self,
        root: TokenNode,
        terminator_token_ids: set[int],
    ) -> bool:
        leaves = self._get_leaf_nodes(root)
        return bool(leaves) and all(self._is_eos(node, terminator_token_ids) for node in leaves)

    def _write_tree(
        self,
        output_path: Path,
        output_data: Dict[str, Any],
    ) -> None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.NamedTemporaryFile(
            "w",
            dir=output_path.parent,
            delete=False,
            suffix=".tmp",
            encoding="utf-8",
        ) as handle:
            json.dump(output_data, handle, ensure_ascii=False, indent=2)
            temp_path = handle.name
        shutil.move(temp_path, output_path)

    def process_tree_file(
        self,
        input_path: Path,
        k: int,
        temperature: float,
        top_p: float,
        min_p: float,
        progress_label: str = "",
    ) -> Dict[str, Any]:
        terminator_token_ids = set(self.terminator_token_ids)
        data, root = self._load_tree_file(input_path)
        source_metadata = data.get("metadata", {})

        source_total_nodes = self._count_nodes(root)
        original_max_depth = self._get_max_depth(root)
        all_eos = self._all_leaves_eos(root, terminator_token_ids)

        progress_prefix = f"{progress_label} " if progress_label else ""
        print(f"\nProcessing {progress_prefix}{input_path.name}")
        print(
            f"  source_total_nodes={source_total_nodes}, "
            f"source_max_depth={original_max_depth}, all_eos={all_eos}"
        )

        pruned_last_depth = False
        pruned_nodes = 0
        resume_max_depth_after_prune = original_max_depth
        preexpand_skipped_reason = ""
        preexpand_stats: Dict[str, Any] = {
            "expanded_nodes": 0,
            "children_added": 0,
            "generate_calls": 0,
            "batch_size_counts": {},
            "terminal_reasons": {},
            "start_total_nodes": self._count_nodes(root),
            "start_leaf_nodes": self._count_leaf_nodes(root),
            "end_total_nodes": self._count_nodes(root),
            "end_leaf_nodes": self._count_leaf_nodes(root),
        }

        if all_eos:
            preexpand_skipped_reason = "all_eos"
        elif original_max_depth > self.target_depth:
            preexpand_skipped_reason = (
                f"source_max_depth_gt_target_depth:{original_max_depth}>{self.target_depth}"
            )

        if preexpand_skipped_reason:
            print(f"  no prune/preexpand: {preexpand_skipped_reason}; walking existing tree")
        else:
            pruned_nodes = self._prune_last_depth(root)
            pruned_last_depth = pruned_nodes > 0
            resume_max_depth_after_prune = self._get_max_depth(root)
            print(
                f"  pruned_last_depth={pruned_last_depth}, "
                f"removed_nodes={pruned_nodes}, resume_max_depth={resume_max_depth_after_prune}"
            )
            preexpand_skipped_reason = "disabled_demand_driven_walk"
            preexpand_stats["end_total_nodes"] = self._count_nodes(root)
            preexpand_stats["end_leaf_nodes"] = self._count_leaf_nodes(root)
            print(
                "  no preexpand: disabled; "
                "non-EOS leaves will be expanded during walk"
            )

        walk_stats_by_seed: Dict[str, Dict[str, Any]] = {}
        for sample_idx in range(self.num_samples):
            run_seed = self.seed + sample_idx
            seed_label = str(run_seed)
            seed_value = _stable_seed(run_seed, input_path.name)
            walk_stats = self._run_walks_for_seed(
                root=root,
                seed_value=seed_value,
                seed_label=seed_label,
                k=k,
                temperature=temperature,
                top_p=top_p,
                min_p=min_p,
                terminator_token_ids=terminator_token_ids,
            )
            walk_stats_by_seed[seed_label] = walk_stats
            print(
                f"  seed={seed_label}: completed={walk_stats['completed_walks']}, "
                f"unique={walk_stats['unique_leaf_hits']}, "
                f"demand_expanded={walk_stats['demand_expanded_nodes']}"
            )

        self._prune_unwalked_paths(root)

        leaf_nodes = self._get_leaf_nodes(root)
        eos_nodes = [node for node in leaf_nodes if self._is_eos(node, terminator_token_ids)]
        total_nodes = self._count_nodes(root)
        max_depth = self._get_max_depth(root)
        avg_depth = sum(node.depth for node in leaf_nodes) / len(leaf_nodes) if leaf_nodes else 0.0
        walk_count_sum = sum(
            sum(_node_extra_dict(node, "_walk_counts").values())
            for node in leaf_nodes
        )

        metadata = dict(source_metadata)
        source_statistics = dict(source_metadata.get("statistics", {}))
        metadata["backend"] = HYBRID_TREE_BACKEND
        metadata["parameters"] = {
            "k": k,
            "temperature": temperature,
            "top_p": top_p,
            "min_p": min_p,
            "max_logprobs": self.max_logprobs,
            "enable_prefix_caching": self.enable_prefix_caching,
            "target_depth": self.target_depth,
            "num_walks": self.num_walks,
            "num_samples": self.num_samples,
            "seeds": [self.seed + idx for idx in range(self.num_samples)],
            "max_batch_size": self.batch_size,
            "walk_batch_size": self.walk_batch_size,
            "walk_policy": "uniform_over_materialized_children_with_demand_expansion",
            "preexpand_policy": "disabled",
            "preexpand_max_leaf_nodes": self.preexpand_max_nodes,
            "preexpand_max_nodes_basis": "leaf_nodes",
            "preexpand_max_nodes_policy": "unused_preexpand_disabled",
            "prune_last_depth_before_resume": True,
            "preexpand_skip_policy": "all_eos_or_source_max_depth_gt_target_or_disabled_demand_driven_walk",
        }
        metadata["statistics"] = {
            "total_nodes": total_nodes,
            "leaf_nodes": len(leaf_nodes),
            "eos_token_nodes": len(eos_nodes),
            "max_depth": max_depth,
            "avg_depth": round(avg_depth, 2),
            "walk_count_sum": walk_count_sum,
            "source_total_nodes": source_total_nodes,
            "source_original_max_depth": original_max_depth,
            "source_statistics": source_statistics,
            "source_all_eos": all_eos,
            "pruned_last_depth": pruned_last_depth,
            "pruned_nodes": pruned_nodes,
            "resume_max_depth_after_prune": resume_max_depth_after_prune,
            "preexpand_skipped_reason": preexpand_skipped_reason,
            "preexpand": preexpand_stats,
            "walks_by_seed": walk_stats_by_seed,
        }
        metadata["status"] = "completed"
        metadata["generated_at"] = datetime.now().isoformat()
        metadata["source_tree_file"] = str(input_path)
        metadata["tree_file_name"] = input_path.name

        output_data = {
            "metadata": metadata,
            "tree": self._node_to_dict(root),
        }

        output_path = self.output_dir / input_path.name
        self._write_tree(output_path, output_data)

        print(
            f"  saved {output_path}: total_nodes={total_nodes}, "
            f"leaf_nodes={len(leaf_nodes)}, max_depth={max_depth}, walks={walk_count_sum}"
        )
        return {
            "file_name": input_path.name,
            "status": "success",
            "filepath": str(output_path),
            "total_nodes": total_nodes,
            "leaf_nodes": len(leaf_nodes),
            "max_depth": max_depth,
            "walk_count_sum": walk_count_sum,
        }

    def _iter_input_files(self) -> List[Path]:
        if self.behaviors_csv:
            input_files: List[Path] = []
            behaviors = load_behaviors(self.behaviors_csv, self.categories)
            for behavior in behaviors:
                for filename in get_tree_file_candidates(behavior):
                    path = self.input_tree_dir / filename
                    if path.exists():
                        input_files.append(path)
                        break
            return input_files

        return [
            path
            for path in sorted(self.input_tree_dir.glob("*.json"))
            if path.name != "batch_summary.json"
        ]

    async def run(
        self,
        k: int,
        temperature: float,
        top_p: float,
        min_p: float,
        start_idx: int = 0,
        max_count: Optional[int] = None,
        resume: bool = False,
    ) -> List[Dict[str, Any]]:
        self._validate_parameters()
        if self.max_logprobs != -1 and k > self.max_logprobs:
            raise ValueError(
                f"k={k} exceeds max_logprobs={self.max_logprobs}. "
                "Increase --max-logprobs or lower --k."
            )

        input_files = self._iter_input_files()
        if max_count is not None:
            input_files = input_files[start_idx:start_idx + max_count]
        else:
            input_files = input_files[start_idx:]

        if not input_files:
            print(f"No JSON tree files found in {self.input_tree_dir}")
            return []

        print(f"\n{'=' * 60}")
        print(f"Starting hybrid uniform-walk tree generation for {len(input_files)} files")
        print(f"Input: {self.input_tree_dir}")
        print(f"Output: {self.output_dir}")
        print(
            f"target_depth={self.target_depth}, num_walks={self.num_walks}, "
            f"num_samples={self.num_samples}, seed={self.seed}"
        )
        print(
            f"k={k}, temperature={temperature}, top_p={top_p}, min_p={min_p}, "
            f"max_logprobs={self.max_logprobs}"
        )
        print(f"{'=' * 60}\n")

        await self.initialize()
        results: List[Dict[str, Any]] = []

        for idx, input_path in enumerate(input_files, start=1):
            output_path = self.output_dir / input_path.name
            if resume and output_path.exists():
                print(f"[{idx}/{len(input_files)}] Skipping existing {output_path.name}")
                results.append({
                    "file_name": input_path.name,
                    "status": "skipped",
                    "filepath": str(output_path),
                })
                continue

            is_valid, error = validate_tree_json(str(input_path))
            if not is_valid:
                print(f"[INVALID] {input_path.name}: {error}")
                results.append({
                    "file_name": input_path.name,
                    "status": "invalid",
                    "error": error,
                })
                continue

            try:
                result = self.process_tree_file(
                    input_path=input_path,
                    k=k,
                    temperature=temperature,
                    top_p=top_p,
                    min_p=min_p,
                    progress_label=f"[{idx}/{len(input_files)}]",
                )
                results.append(result)
            except torch.cuda.OutOfMemoryError as exc:
                print(f"[OOM] {input_path.name} - reloading model")
                results.append({
                    "file_name": input_path.name,
                    "status": "oom_skipped",
                    "error": str(exc),
                })
                self.cleanup()
                await self.initialize()
            except RuntimeError as exc:
                if "out of memory" in str(exc).lower():
                    print(f"[OOM] {input_path.name} - reloading model")
                    results.append({
                        "file_name": input_path.name,
                        "status": "oom_skipped",
                        "error": str(exc),
                    })
                    self.cleanup()
                    await self.initialize()
                else:
                    print(f"[ERROR] {input_path.name}: {exc}")
                    results.append({
                        "file_name": input_path.name,
                        "status": "error",
                        "error": str(exc),
                    })
            except Exception as exc:
                print(f"[ERROR] {input_path.name}: {exc}")
                results.append({
                    "file_name": input_path.name,
                    "status": "error",
                    "error": str(exc),
                })
            finally:
                torch.cuda.empty_cache()

        summary_path = self.output_dir / "batch_summary.json"
        self._write_tree(
            summary_path,
            {
                "backend": HYBRID_TREE_BACKEND,
                "generated_at": datetime.now().isoformat(),
                "input_tree_dir": str(self.input_tree_dir),
                "output_dir": str(self.output_dir),
                "results": results,
            },
        )

        success_count = sum(1 for result in results if result["status"] == "success")
        skipped_count = sum(1 for result in results if result["status"] == "skipped")
        print(f"\n{'=' * 60}")
        print("Hybrid uniform-walk tree generation complete")
        print(f"Success: {success_count}/{len(results)}")
        if skipped_count:
            print(f"Skipped: {skipped_count}")
        print(f"Summary: {summary_path}")
        print(f"{'=' * 60}")

        self.cleanup()
        return results


async def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(
        description="Hybrid vLLM tree resume + seed-indexed uniform random walks"
    )
    parser.add_argument("--input-tree-dir", required=True, help="Existing tree JSON directory")
    parser.add_argument("--output-dir", required=True, help="Output directory for walked trees")
    parser.add_argument("--behaviors-csv", default=None, help="Optional behaviors CSV for tree file ordering")
    parser.add_argument("--categories", nargs="+", default=None)
    parser.add_argument("--model-path", required=True, help="Path to model")
    parser.add_argument("--model-type", default="llama2")
    parser.add_argument("--dtype", default="float16")
    parser.add_argument("--k", type=int, default=0)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top-p", type=float, default=0.9)
    parser.add_argument("--min-p", type=float, default=0.0)
    parser.add_argument(
        "--batch-size",
        "--max-batch-size",
        dest="batch_size",
        type=int,
        default=128,
        help="Maximum vLLM batch size for preexpand and demand expansion",
    )
    parser.add_argument("--num-gpus", type=int, default=2)
    parser.add_argument("--max-model-len", type=int, default=2048)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.7)
    parser.add_argument("--max-logprobs", type=int, default=200)
    parser.add_argument("--target-depth", type=int, default=DEFAULT_TARGET_DEPTH)
    parser.add_argument("--num-walks", type=int, default=DEFAULT_NUM_WALKS)
    parser.add_argument("--num-samples", type=int, default=1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--walk-batch-size", type=int, default=DEFAULT_NUM_WALKS)
    parser.add_argument(
        "--preexpand-min-depth",
        type=int,
        default=None,
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--preexpand-max-nodes",
        type=int,
        default=DEFAULT_PREEXPAND_MAX_NODES,
        help="Unused legacy option; preexpand is disabled",
    )
    parser.add_argument("--start-idx", type=int, default=0)
    parser.add_argument("--max-count", type=int, default=None)
    parser.add_argument("--resume", action="store_true", help="Skip output files that already exist")
    parser.add_argument(
        "--allow-in-place",
        action="store_true",
        help="Allow output-dir to be the same as input-tree-dir",
    )
    parser.add_argument(
        "--disable-prefix-caching",
        action="store_true",
        help="Disable vLLM automatic prefix caching",
    )
    parser.add_argument(
        "--system-prompt-suffix",
        type=str,
        default=None,
        help="Path to file with additional system prompt text",
    )
    args = parser.parse_args()

    input_dir = Path(args.input_tree_dir)
    output_dir = Path(args.output_dir)
    if input_dir.resolve() == output_dir.resolve() and not args.allow_in_place:
        raise ValueError(
            "--output-dir must differ from --input-tree-dir unless --allow-in-place is set"
        )

    if args.system_prompt_suffix:
        with open(args.system_prompt_suffix, "r", encoding="utf-8") as handle:
            set_system_prompt_suffix(handle.read().strip())

    explorer = HybridRandomWalkTreeExplorerVLLM(
        model_path=args.model_path,
        model_type=args.model_type,
        dtype=args.dtype,
        output_dir=str(output_dir),
        input_tree_dir=str(input_dir),
        behaviors_csv=args.behaviors_csv,
        categories=args.categories,
        num_gpus=args.num_gpus,
        max_model_len=args.max_model_len,
        gpu_memory_utilization=args.gpu_memory_utilization,
        batch_size=args.batch_size,
        max_depth=MAX_DEPTH,
        max_logprobs=args.max_logprobs,
        enable_prefix_caching=not args.disable_prefix_caching,
        target_depth=args.target_depth,
        num_walks=args.num_walks,
        num_samples=args.num_samples,
        seed=args.seed,
        walk_batch_size=args.walk_batch_size,
        preexpand_max_nodes=args.preexpand_max_nodes,
    )

    results = await explorer.run(
        k=args.k,
        temperature=args.temperature,
        top_p=args.top_p,
        min_p=args.min_p,
        start_idx=args.start_idx,
        max_count=args.max_count,
        resume=args.resume,
    )

    failed = [result for result in results if result["status"] not in {"success", "skipped"}]
    if failed:
        print(f"[ERROR] Failed files: {[result['file_name'] for result in failed]}")
        sys.exit(1)

    print(f"GENERATED_TREE_DIR={args.output_dir}")


if __name__ == "__main__":
    asyncio.run(main())
