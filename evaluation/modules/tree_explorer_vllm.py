"""vLLM-backed tree explorer using processed logprobs."""

import os
import sys

if __name__ == "__main__":
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import asyncio
import csv
import hashlib
import json
import math
import shutil
import tempfile
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import torch
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams

try:
    from .token_node import TokenNode
    from .model_utils import set_system_prompt_suffix
except ImportError:
    from token_node import TokenNode
    from model_utils import set_system_prompt_suffix

MAX_LEAVES = 1000
MAX_DEPTH = 512
RUN_SEPARATOR = "=" * 60


def _get_terminator_token_ids(tokenizer: AutoTokenizer, model_type: str) -> List[int]:
    """Return model terminator token ids used to end assistant generation."""
    terminator_ids: List[int] = []
    eos_token_id = getattr(tokenizer, "eos_token_id", None)

    if isinstance(eos_token_id, int):
        terminator_ids.append(eos_token_id)
    elif isinstance(eos_token_id, (list, tuple)):
        terminator_ids.extend(int(token_id) for token_id in eos_token_id)

    if model_type == "llama3":
        terminator_ids.extend([128001, 128008, 128009])

    if model_type == "qwen":
        endoftext_token_id = tokenizer.convert_tokens_to_ids("<|endoftext|>")
        if isinstance(endoftext_token_id, int) and endoftext_token_id >= 0:
            terminator_ids.append(endoftext_token_id)

    if model_type == "gemma":
        end_of_turn_id = tokenizer.convert_tokens_to_ids("<end_of_turn>")
        if isinstance(end_of_turn_id, int) and end_of_turn_id >= 0:
            terminator_ids.append(end_of_turn_id)

    return list(dict.fromkeys(terminator_ids))


def validate_tree_json(filepath: str) -> Tuple[bool, Optional[str]]:
    """Validate the minimal tree JSON shape used by continuation scripts."""
    try:
        with open(filepath, "r", encoding="utf-8") as handle:
            data = json.load(handle)

        if "metadata" not in data:
            return False, "Missing 'metadata' key"
        if "tree" not in data:
            return False, "Missing 'tree' key"

        metadata = data["metadata"]
        for key in ["prompt", "parameters", "statistics"]:
            if key not in metadata:
                return False, f"Missing metadata field: {key}"

        tree = data["tree"]
        if "text" not in tree:
            return False, "Tree missing 'text' field"
        if "children" not in tree:
            return False, "Tree missing 'children' field"

        stats = metadata["statistics"]
        if stats.get("total_nodes", 0) < 1:
            return False, "Invalid statistics: total_nodes < 1"
        if stats.get("leaf_nodes", 0) < 1:
            return False, "Invalid statistics: leaf_nodes < 1"

        return True, None
    except json.JSONDecodeError as exc:
        return False, f"JSON parse error: {exc}"
    except Exception as exc:
        return False, f"Validation error: {exc}"


def tree_stats_are_complete(stats: Dict[str, Any]) -> bool:
    """Return whether an existing tree should be accepted for resume."""
    leaf_count = stats.get("leaf_nodes", 0)
    eos_count = stats.get("eos_token_nodes", 0)
    reached_leaf_budget = leaf_count == MAX_LEAVES
    all_leaves_are_eos = eos_count == leaf_count and leaf_count > 0
    return reached_leaf_budget or all_leaves_are_eos


def get_tree_output_filename(behavior: Dict[str, Any]) -> str:
    """Return the preferred tree filename for a behavior row."""
    tree_file_name = str(behavior.get("TreeFileName", "")).strip()
    if tree_file_name:
        return tree_file_name

    behavior_id = str(behavior.get("BehaviorID", "")).strip()
    if behavior_id:
        return f"{behavior_id}.json"

    return ""


def get_previous_tree_output_filenames(behavior: Dict[str, Any]) -> List[str]:
    """Return legacy filename variants for backward-compatible resume."""
    tree_file_name = str(behavior.get("TreeFileName", "")).strip()
    if not tree_file_name:
        return []

    prompt_id = str(behavior.get("PromptID", "")).strip()
    prefix_id = str(behavior.get("PrefixID", "")).strip()
    path = Path(tree_file_name)
    stem = path.stem
    suffix = path.suffix or ".json"

    candidates: List[str] = []
    prefixed_pattern = f"{prefix_id}__{prompt_id}_"
    no_prefix_pattern = f"{prompt_id}_"

    if prompt_id and prefix_id and stem.startswith(prefixed_pattern):
        output_name = stem[len(prefixed_pattern):]
        if output_name:
            candidates.append(f"{output_name}_{prefix_id}__{prompt_id}{suffix}")

    if prompt_id and stem.startswith(no_prefix_pattern):
        output_name = stem[len(no_prefix_pattern):]
        if output_name:
            candidates.append(f"{output_name}_{prompt_id}{suffix}")

    return candidates


def get_tree_file_candidates(behavior: Dict[str, Any]) -> List[str]:
    """Return preferred and legacy tree filenames for a behavior."""
    candidates: List[str] = []

    preferred = get_tree_output_filename(behavior)
    if preferred:
        candidates.append(preferred)

    behavior_id = str(behavior.get("BehaviorID", "")).strip()
    legacy = f"{behavior_id}.json" if behavior_id else ""
    if legacy and legacy not in candidates:
        candidates.append(legacy)

    for previous_name in get_previous_tree_output_filenames(behavior):
        if previous_name and previous_name not in candidates:
            candidates.append(previous_name)

    return candidates


def resolve_existing_tree_file(tree_dir: Path, behavior: Dict[str, Any]) -> Optional[Path]:
    """Find an existing tree file for a behavior, supporting legacy names."""
    for filename in get_tree_file_candidates(behavior):
        candidate = tree_dir / filename
        if candidate.exists():
            return candidate
    return None


def load_behaviors(csv_path: str, categories: Optional[List[str]] = None) -> List[Dict[str, Any]]:
    """Load HarmBench-style behavior rows."""
    selected_categories = categories or ["contextual", "standard"]
    behaviors: List[Dict[str, Any]] = []

    with open(csv_path, "r", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            if row.get("FunctionalCategory", "standard") in selected_categories:
                behaviors.append({
                    "Behavior": row.get("Behavior", ""),
                    "ContextString": row.get("ContextString", ""),
                    "BehaviorID": row.get("BehaviorID", ""),
                    "PrefixID": row.get("PrefixID", ""),
                    "PromptID": row.get("PromptID", ""),
                    "TreeFileName": row.get("TreeFileName", ""),
                    "SemanticCategory": row.get("SemanticCategory", ""),
                    "FunctionalCategory": row.get("FunctionalCategory", "standard"),
                })

    return behaviors


def build_prompt(behavior: str, context_str: Optional[str] = None) -> str:
    """Build the behavior prompt before model-specific chat formatting."""
    if context_str and context_str.strip():
        return f"{context_str}\n\n---\n\n{behavior}"
    return behavior


def validate_trees_for_behaviors(
    tree_dir: str,
    behaviors: List[Dict[str, Any]],
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """Validate tree JSON files using this module's MAX_LEAVES threshold."""
    tree_dir_path = Path(tree_dir)
    valid_behaviors: List[Dict[str, Any]] = []
    missing_behaviors: List[Dict[str, Any]] = []

    for behavior in behaviors:
        behavior_id = behavior.get("BehaviorID", "")
        tree_file = resolve_existing_tree_file(tree_dir_path, behavior)

        if tree_file is None:
            missing_behaviors.append(behavior)
            continue

        is_valid, error = validate_tree_json(str(tree_file))
        if is_valid:
            with open(tree_file, "r", encoding="utf-8") as handle:
                data = json.load(handle)
            metadata = data.get("metadata", {})
            stats = metadata.get("statistics", {})

            if tree_stats_are_complete(stats):
                valid_behaviors.append(behavior)
            else:
                missing_behaviors.append(behavior)
        else:
            print(f"[INVALID] {behavior_id}: {error}")
            missing_behaviors.append(behavior)

    return valid_behaviors, missing_behaviors


class TreeExplorerVLLM:
    """Tree explorer that expands one token at a time using vLLM logprobs."""

    def __init__(
        self,
        model_path: str,
        model_type: str = "llama2",
        dtype: str = "float16",
        output_dir: str = "generated_trees",
        num_gpus: int = 2,
        max_model_len: int = 2048,
        gpu_memory_utilization: float = 0.7,
        batch_size: int = 128,
        max_depth: int = MAX_DEPTH,
        max_logprobs: int = 50,
        enable_prefix_caching: bool = True,
    ):
        self.model_path = model_path
        self.model_type = model_type
        self.dtype = dtype
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.num_gpus = num_gpus
        self.max_model_len = max_model_len
        self.gpu_memory_utilization = gpu_memory_utilization
        self.batch_size = batch_size
        self.max_depth = max_depth
        self.max_logprobs = max_logprobs
        self.enable_prefix_caching = enable_prefix_caching

        self.llm: Optional[LLM] = None
        self.tokenizer: Optional[AutoTokenizer] = None
        self.terminator_token_ids: List[int] = []

    async def initialize(self) -> None:
        """Load tokenizer and vLLM engine."""
        print(f"Loading tokenizer from {self.model_path}...")
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_path,
            use_fast=True,
            trust_remote_code=True,
        )
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self.terminator_token_ids = _get_terminator_token_ids(
            self.tokenizer, self.model_type
        )

        print("Loading vLLM engine...")
        self.llm = LLM(
            model=self.model_path,
            dtype=self.dtype,
            tensor_parallel_size=self.num_gpus,
            max_model_len=self.max_model_len,
            gpu_memory_utilization=self.gpu_memory_utilization,
            trust_remote_code=True,
            enable_prefix_caching=self.enable_prefix_caching,
            logprobs_mode="processed_logprobs",
            max_logprobs=self.max_logprobs,
        )
        print("Initialization complete!")

    def cleanup(self) -> None:
        """Release GPU memory."""
        print("Cleaning up tree explorer vLLM...")
        if self.llm:
            del self.llm
            self.llm = None
        if self.tokenizer:
            del self.tokenizer
            self.tokenizer = None
        import gc

        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
        print("Cleanup complete!")

    def _decode_token(self, token_id: int) -> str:
        """Decode a single token in the same way as the HF explorer."""
        assert self.tokenizer is not None
        if self.model_type == "llama2":
            raw_token = self.tokenizer.convert_ids_to_tokens([token_id])[0]
            return raw_token.replace("▁", " ")
        return self.tokenizer.decode([token_id], skip_special_tokens=False)

    def _get_leaf_nodes(self, node: TokenNode) -> List[TokenNode]:
        if not node.children:
            return [node]
        leaves: List[TokenNode] = []
        for child in node.children.values():
            leaves.extend(self._get_leaf_nodes(child))
        return leaves

    def _count_nodes(self, node: TokenNode) -> int:
        count = 1
        for child in node.children.values():
            count += self._count_nodes(child)
        return count

    def _get_max_depth(self, node: TokenNode) -> int:
        if not node.children:
            return node.depth
        return max(self._get_max_depth(child) for child in node.children.values())

    def _prune_low_probability_leaves(
        self,
        root: TokenNode,
    ) -> int:
        """Prune leaves in reverse DFS order at the deepest depth.

        Reverse DFS order keeps the first greedy-style path stable. Removing an
        only child turns its parent into a leaf, so the effective leaf count does
        not decrease in that case.
        """
        new_leaves = self._get_leaf_nodes(root)
        if len(new_leaves) <= MAX_LEAVES:
            return len(new_leaves)

        max_leaf_depth = max(n.depth for n in new_leaves)

        targets = [n for n in new_leaves if n.depth == max_leaf_depth]
        targets.reverse()

        to_remove = len(new_leaves) - MAX_LEAVES
        for node in targets:
            if to_remove <= 0:
                break
            parent = node.parent
            if parent is None:
                continue
            token_key = node.token_id
            if token_key in parent.children:
                is_only_child = len(parent.children) == 1
                del parent.children[token_key]
                if not is_only_child:
                    to_remove -= 1

        result_leaves = self._get_leaf_nodes(root)
        print(f"  [PRUNE] depth {max_leaf_depth}: "
              f"{len(new_leaves)} -> {len(result_leaves)} leaves")
        return len(result_leaves)

    def _iter_logprob_items(self, logprobs_one_position: Any) -> Iterable[Tuple[int, Any]]:
        """Yield token_id, logprob_info pairs from vLLM logprob containers."""
        if logprobs_one_position is None:
            return []
        if hasattr(logprobs_one_position, "items"):
            return list(logprobs_one_position.items())
        return []

    def _extract_candidates(self, sample_logprobs: Any) -> List[Tuple[int, float, Optional[int], Optional[str]]]:
        """Extract processed candidates from a vLLM one-token decode."""
        if sample_logprobs is None:
            return []
        try:
            first_position = sample_logprobs[0]
        except Exception:
            return []
        if first_position is None:
            return []

        candidates: List[Tuple[int, float, Optional[int], Optional[str]]] = []
        for token_id, info in self._iter_logprob_items(first_position):
            logprob = getattr(info, "logprob", None)
            rank = getattr(info, "rank", None)
            decoded_token = getattr(info, "decoded_token", None)
            if logprob is None or not math.isfinite(float(logprob)):
                continue
            candidates.append((int(token_id), float(logprob), rank, decoded_token))

        if not candidates:
            return []

        ranked = [c for c in candidates if c[2] is not None]
        if ranked:
            ranked.sort(key=lambda item: (item[2], -item[1], item[0]))
            return ranked

        candidates.sort(key=lambda item: (-item[1], item[0]))
        return candidates

    def _build_sampling_params(
        self,
        k: int,
        temperature: float,
        top_p: float,
        min_p: float,
    ) -> SamplingParams:
        if self.max_logprobs == -1:
            request_logprobs = -1
        elif k > 0:
            request_logprobs = min(k, self.max_logprobs)
        else:
            request_logprobs = self.max_logprobs
        top_k = k if k > 0 else 0
        return SamplingParams(
            n=1,
            max_tokens=1,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            min_p=min_p,
            logprobs=request_logprobs,
            detokenize=True,
            skip_special_tokens=False,
            ignore_eos=True,
        )

    def _expand_batch(
        self,
        nodes: Sequence[TokenNode],
        k: int,
        temperature: float,
        top_p: float,
        min_p: float,
    ) -> None:
        """Expand a batch of leaf nodes by one token."""
        assert self.llm is not None
        assert self.tokenizer is not None

        prompts = []
        for node in nodes:
            prefix = node.get_token_prefix(self.tokenizer, self.model_type)
            prompts.append({"prompt_token_ids": prefix})

        sampling_params = self._build_sampling_params(k, temperature, top_p, min_p)
        outputs = self.llm.generate(prompts, sampling_params=sampling_params)

        for node, output in zip(nodes, outputs):
            if not output.outputs:
                continue
            candidates = self._extract_candidates(output.outputs[0].logprobs)
            for token_id, logprob, _rank, decoded_token in candidates:
                token_text = decoded_token if decoded_token is not None else self._decode_token(token_id)
                prob_value = float(math.exp(logprob))
                child_node = TokenNode(
                    id=os.urandom(4).hex(),
                    token_id=token_id,
                    text=token_text,
                    prob=prob_value,
                    score=prob_value,
                    depth=node.depth + 1,
                    parent=node,
                    children={},
                )
                node.children[token_id] = child_node

    async def generate_tree(
        self,
        prompt: str,
        exploration_depths: Optional[List[int]] = None,
        k: int = 5,
        temperature: float = 0.7,
        top_p: float = 0.9,
        min_p: float = 0.0,
        output_filename: Optional[str] = None,
        extra_metadata: Optional[Dict[str, Any]] = None,
    ) -> Tuple[str, TokenNode]:
        """Generate a token tree using one-step vLLM expansions."""
        del exploration_depths  # Kept only for CLI compatibility.

        print(f"\nGenerating tree for prompt: '{prompt[:50]}...'")
        print(f"k={k}, temperature={temperature}, top_p={top_p}, min_p={min_p}")

        if self.max_logprobs != -1 and k > self.max_logprobs:
            raise ValueError(
                f"k={k} exceeds max_logprobs={self.max_logprobs}. "
                "Increase --max-logprobs or lower --k."
            )

        root = TokenNode(
            id="root",
            token_id=-1,
            text=prompt,
            prob=1.0,
            score=1.0,
            depth=0,
            parent=None,
            children={},
        )
        terminator_token_ids = set(self.terminator_token_ids)

        phase = 0
        while True:
            phase += 1
            prev_leaves = [
                n for n in self._get_leaf_nodes(root)
                if n.token_id not in terminator_token_ids and n.depth < self.max_depth
            ]

            if not prev_leaves:
                max_d = self._get_max_depth(root)
                if max_d >= self.max_depth:
                    print(f"Depth {phase}: done (max depth {self.max_depth})")
                else:
                    print(f"Depth {phase}: done (all EOS)")
                break

            if phase % 10 == 0:
                print(f"Depth {phase}: {len(prev_leaves)} leaves")

            for start in range(0, len(prev_leaves), self.batch_size):
                batch = prev_leaves[start:start + self.batch_size]
                self._expand_batch(
                    nodes=batch,
                    k=k,
                    temperature=temperature,
                    top_p=top_p,
                    min_p=min_p,
                )

            leaves_before_prune = len(self._get_leaf_nodes(root))
            new_leaf_count = self._prune_low_probability_leaves(root)
            if leaves_before_prune > MAX_LEAVES:
                print(f"Depth {phase}: depth-pruned {leaves_before_prune} -> {new_leaf_count} leaves, stopping.")
                break

        tree_dict = root.to_dict()
        leaf_nodes = self._get_leaf_nodes(root)
        eos_nodes = [n for n in leaf_nodes if n.token_id in terminator_token_ids]
        avg_depth = sum(n.depth for n in leaf_nodes) / len(leaf_nodes) if leaf_nodes else 0.0
        total_nodes = self._count_nodes(root)
        max_depth = self._get_max_depth(root)

        metadata = {
            "prompt": prompt,
            "model_path": self.model_path,
            "model_type": self.model_type,
            "backend": "vllm_processed_logprobs",
            "parameters": {
                "k": k,
                "temperature": temperature,
                "top_p": top_p,
                "min_p": min_p,
                "max_logprobs": self.max_logprobs,
                "enable_prefix_caching": self.enable_prefix_caching,
            },
            "statistics": {
                "total_nodes": total_nodes,
                "leaf_nodes": len(leaf_nodes),
                "eos_token_nodes": len(eos_nodes),
                "max_depth": max_depth,
                "avg_depth": round(avg_depth, 2),
            },
            "status": "completed",
            "generated_at": datetime.now().isoformat(),
        }
        if extra_metadata:
            metadata.update(extra_metadata)

        output_data = {
            "metadata": metadata,
            "tree": tree_dict,
        }

        if output_filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            prompt_hash = hashlib.md5(prompt.encode()).hexdigest()[:8]
            output_filename = f"tree_{timestamp}_{prompt_hash}.json"

        output_path = self.output_dir / output_filename
        print(f"\nSaving tree to {output_path}...")
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

        print(f"  Total nodes: {total_nodes}")
        print(f"  Leaf nodes: {len(leaf_nodes)} (EOS: {len(eos_nodes)})")
        print(f"  Max depth: {max_depth}, Avg depth: {avg_depth:.1f}")
        return str(output_path), root


async def run_tree_generation(
    behaviors: List[Dict[str, Any]],
    model_path: str,
    model_type: str,
    dtype: str,
    output_dir: str,
    exploration_depths: Optional[List[int]] = None,
    k: int = 5,
    temperature: float = 0.7,
    top_p: float = 0.9,
    min_p: float = 0.0,
    batch_size: int = 128,
    num_gpus: int = 2,
    max_model_len: int = 2048,
    gpu_memory_utilization: float = 0.7,
    max_depth: int = MAX_DEPTH,
    max_logprobs: int = 50,
    enable_prefix_caching: bool = True,
) -> List[Dict[str, Any]]:
    """Run vLLM tree generation for a list of behaviors."""
    print(f"\n{RUN_SEPARATOR}")
    print(f"Starting vLLM tree generation for {len(behaviors)} behaviors")
    print(f"Model: {model_path}")
    print(f"Output: {output_dir}")
    print(f"k={k}, batch_size={batch_size}, max_logprobs={max_logprobs}")
    print(f"Prefix caching: {enable_prefix_caching}")
    print(f"{RUN_SEPARATOR}\n")

    output_dir_path = Path(output_dir)
    valid_behaviors, missing_behaviors = validate_trees_for_behaviors(
        str(output_dir_path), behaviors
    )

    if not missing_behaviors:
        print(f"\n[INFO] All {len(behaviors)} trees already exist and are valid. Skipping model load.")
        return [
            {
                "behavior_id": b.get("BehaviorID"),
                "status": "success",
                "filepath": str(resolve_existing_tree_file(output_dir_path, b)),
                "skipped": True,
            }
            for b in behaviors
        ]

    print(f"[INFO] Valid: {len(valid_behaviors)}, Missing/Invalid: {len(missing_behaviors)}")

    results = [
        {
            "behavior_id": b.get("BehaviorID"),
            "status": "success",
            "filepath": str(resolve_existing_tree_file(output_dir_path, b)),
            "skipped": True,
        }
        for b in valid_behaviors
    ]

    explorer = TreeExplorerVLLM(
        model_path=model_path,
        model_type=model_type,
        dtype=dtype,
        output_dir=output_dir,
        num_gpus=num_gpus,
        max_model_len=max_model_len,
        gpu_memory_utilization=gpu_memory_utilization,
        batch_size=batch_size,
        max_depth=max_depth,
        max_logprobs=max_logprobs,
        enable_prefix_caching=enable_prefix_caching,
    )
    await explorer.initialize()

    for i, behavior_dict in enumerate(missing_behaviors):
        behavior_id = behavior_dict.get("BehaviorID", f"behavior_{i}")
        behavior = behavior_dict.get("Behavior", "")
        context_str = behavior_dict.get("ContextString", "")
        existing_tree_file = resolve_existing_tree_file(output_dir_path, behavior_dict)
        output_filename = (
            existing_tree_file.name if existing_tree_file else get_tree_output_filename(behavior_dict)
        )

        print(f"\n[{i + 1}/{len(missing_behaviors)}] Processing: {behavior_id}")
        print(f"Behavior: {behavior[:80]}...")

        prompt = build_prompt(behavior, context_str)
        extra_metadata = {
            "behavior_id": behavior_id,
            "prefix_id": behavior_dict.get("PrefixID", ""),
            "prompt_id": behavior_dict.get("PromptID", ""),
            "tree_file_name": output_filename,
        }

        try:
            filepath, _ = await explorer.generate_tree(
                prompt=prompt,
                exploration_depths=exploration_depths,
                k=k,
                temperature=temperature,
                top_p=top_p,
                min_p=min_p,
                output_filename=output_filename,
                extra_metadata=extra_metadata,
            )
            results.append({
                "behavior_id": behavior_id,
                "status": "success",
                "filepath": filepath,
            })
        except torch.cuda.OutOfMemoryError as exc:
            print(f"[OOM] {behavior_id} - skipping and reloading model...")
            results.append({
                "behavior_id": behavior_id,
                "status": "oom_skipped",
                "error": str(exc),
            })
            explorer.cleanup()
            await explorer.initialize()
            continue
        except RuntimeError as exc:
            if "out of memory" in str(exc).lower():
                print(f"[OOM] {behavior_id} - skipping and reloading model...")
                results.append({
                    "behavior_id": behavior_id,
                    "status": "oom_skipped",
                    "error": str(exc),
                })
                explorer.cleanup()
                await explorer.initialize()
                continue
            results.append({
                "behavior_id": behavior_id,
                "status": "error",
                "error": str(exc),
            })
            print(f"Error: {exc}")
        except Exception as exc:
            results.append({
                "behavior_id": behavior_id,
                "status": "error",
                "error": str(exc),
            })
            print(f"Error: {exc}")

        torch.cuda.empty_cache()

    success_count = sum(1 for r in results if r["status"] == "success")
    oom_count = sum(1 for r in results if r["status"] == "oom_skipped")
    error_count = sum(1 for r in results if r["status"] == "error")
    print(f"\n{RUN_SEPARATOR}")
    print("Tree generation complete!")
    print(f"Success: {success_count}/{len(results)}")
    if oom_count > 0:
        print(f"OOM skipped: {oom_count}")
    if error_count > 0:
        print(f"Errors: {error_count}")
    print(RUN_SEPARATOR)

    explorer.cleanup()
    return results


async def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="vLLM tree exploration for behaviors")
    parser.add_argument("--behaviors-csv", required=True, help="Path to behaviors CSV")
    parser.add_argument("--model-path", required=True, help="Path to model")
    parser.add_argument("--model-type", default="llama2", help="Model type")
    parser.add_argument("--dtype", default="float16", help="Model dtype")
    parser.add_argument("--output-dir", required=True, help="Output directory")
    parser.add_argument("--exploration-depths", nargs="+", type=int, default=[3, 3, 3, 3, 3, 3, 2])
    parser.add_argument("--k", type=int, default=5)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top-p", type=float, default=0.9)
    parser.add_argument("--min-p", type=float, default=0.0)
    parser.add_argument("--categories", nargs="+", default=["contextual", "standard"])
    parser.add_argument("--start-idx", type=int, default=0)
    parser.add_argument("--max-count", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--num-gpus", type=int, default=2)
    parser.add_argument("--max-model-len", type=int, default=2048)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.7)
    parser.add_argument("--max-depth", type=int, default=MAX_DEPTH)
    parser.add_argument("--max-logprobs", type=int, default=200)
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

    if args.system_prompt_suffix:
        with open(args.system_prompt_suffix, "r", encoding="utf-8") as handle:
            suffix_text = handle.read().strip()
        set_system_prompt_suffix(suffix_text)

    behaviors = load_behaviors(args.behaviors_csv, args.categories)
    if args.max_count:
        behaviors = behaviors[args.start_idx:args.start_idx + args.max_count]
    else:
        behaviors = behaviors[args.start_idx:]

    total_expected = len(behaviors)
    print(f"Processing {total_expected} behaviors")

    results = await run_tree_generation(
        behaviors=behaviors,
        model_path=args.model_path,
        model_type=args.model_type,
        dtype=args.dtype,
        output_dir=args.output_dir,
        exploration_depths=args.exploration_depths,
        k=args.k,
        temperature=args.temperature,
        top_p=args.top_p,
        min_p=args.min_p,
        batch_size=args.batch_size,
        num_gpus=args.num_gpus,
        max_model_len=args.max_model_len,
        gpu_memory_utilization=args.gpu_memory_utilization,
        max_depth=args.max_depth,
        max_logprobs=args.max_logprobs,
        enable_prefix_caching=not args.disable_prefix_caching,
    )

    success_count = sum(1 for r in results if r["status"] == "success")
    failed_count = total_expected - success_count

    if failed_count > 0:
        failed_behaviors = [r["behavior_id"] for r in results if r["status"] != "success"]
        print(f"\n{RUN_SEPARATOR}")
        print(f"[ERROR] Tree generation failed for {failed_count}/{total_expected} behaviors!")
        print(f"Failed behaviors: {failed_behaviors}")
        print("Aborting pipeline - continuation and classifier will not run.")
        print(RUN_SEPARATOR)
        sys.exit(1)

    print(f"GENERATED_TREE_DIR={args.output_dir}")


if __name__ == "__main__":
    asyncio.run(main())
