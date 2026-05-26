"""Generate continuations from hybrid random-walk tree `walk_counts`.

Unlike random_continuation.py, this module does not simulate walks. It expects
hybrid tree JSON files where walked leaves contain seed-indexed walk_counts,
then emits one continuation row per effective walk visit.
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import sys
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams

if __name__ == "__main__":
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

try:
    from .model_utils import get_template, set_system_prompt_suffix
    from .tree_explorer_vllm import _get_terminator_token_ids
except ImportError:
    from model_utils import get_template, set_system_prompt_suffix
    from tree_explorer_vllm import _get_terminator_token_ids


def _walk_counts(node: Dict[str, Any]) -> Dict[str, int]:
    raw = node.get("walk_counts", {})
    if not isinstance(raw, dict):
        return {}
    return {str(seed): int(count) for seed, count in raw.items() if int(count) > 0}


def _seed_sample_index_map(metadata: Dict[str, Any]) -> Dict[str, int]:
    parameters = metadata.get("parameters", {})
    seeds = parameters.get("seeds", [])
    if isinstance(seeds, list) and seeds:
        return {str(seed): idx for idx, seed in enumerate(seeds)}
    return {}


def collect_walk_leaves(
    node: Dict[str, Any],
    current_tokens: Optional[List[int]] = None,
    current_texts: Optional[List[str]] = None,
    current_prob: float = 1.0,
) -> List[Dict[str, Any]]:
    if current_tokens is None:
        current_tokens = []
    if current_texts is None:
        current_texts = []

    if node.get("id") != "root":
        current_tokens = current_tokens + [int(node.get("token_id", -1))]
        current_texts = current_texts + [node.get("text", "")]
        current_prob *= float(node.get("prob", 1.0))

    children = node.get("children", [])
    if children:
        rows: List[Dict[str, Any]] = []
        for child in children:
            rows.extend(
                collect_walk_leaves(
                    child,
                    current_tokens=current_tokens,
                    current_texts=current_texts,
                    current_prob=current_prob,
                )
            )
        return rows

    counts = _walk_counts(node)
    if not counts:
        return []

    return [
        {
            "leaf_id": node.get("id", ""),
            "token_ids": current_tokens,
            "texts": current_texts,
            "path_probability": current_prob,
            "leaf_token_id": int(node.get("token_id", -1)),
            "walk_counts": counts,
            "terminal_reason": node.get("terminal_reason", ""),
        }
    ]


class HybridWalkContinuationGenerator:
    """Generate continuation rows from seed-indexed hybrid walk leaves."""

    def __init__(
        self,
        model_path: str,
        model_type: str = "llama2",
        dtype: str = "float16",
        num_gpus: int = 2,
        max_model_len: int = 2048,
        gpu_memory_utilization: float = 0.7,
    ):
        self.model_path = model_path
        self.model_type = model_type
        self.dtype = dtype
        self.num_gpus = num_gpus
        self.max_model_len = max_model_len
        self.gpu_memory_utilization = gpu_memory_utilization

        self.llm = None
        self.tokenizer = None
        self.template = None
        self.terminator_token_ids: List[int] = []

    def initialize(self) -> None:
        print(f"Loading continuation model from {self.model_path}...")
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_path,
            use_fast=True,
            trust_remote_code=True,
        )
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self.terminator_token_ids = _get_terminator_token_ids(
            self.tokenizer,
            self.model_type,
        )
        self.llm = LLM(
            model=self.model_path,
            dtype=self.dtype,
            tensor_parallel_size=self.num_gpus,
            max_model_len=self.max_model_len,
            gpu_memory_utilization=self.gpu_memory_utilization,
            trust_remote_code=True,
            enable_prefix_caching=True,
        )
        self.template = get_template(
            model_name_or_path=self.model_path,
            model_type=self.model_type,
        )
        print("Continuation model loaded!")

    def cleanup(self) -> None:
        print("Cleaning up continuation model...")
        if self.llm:
            del self.llm
            self.llm = None
        if self.tokenizer:
            del self.tokenizer
            self.tokenizer = None

        import gc
        import torch

        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
        print("Cleanup complete!")

    def _build_job(
        self,
        file_name: str,
        metadata: Dict[str, Any],
        leaf: Dict[str, Any],
        seed_label: str,
        base_sample_idx: int,
        visit_sample_idx: int,
        walk_count: int,
        num_samples: int,
    ) -> Dict[str, Any]:
        output_sample_idx = base_sample_idx + visit_sample_idx * num_samples
        return {
            "file_name": file_name,
            "leaf_id": leaf["leaf_id"],
            "prompt": metadata.get("prompt", ""),
            "token_ids": leaf["token_ids"],
            "texts": leaf["texts"],
            "depth": len(leaf["texts"]),
            "path_probability": leaf["path_probability"],
            "leaf_token_id": leaf["leaf_token_id"],
            "sample_idx": output_sample_idx,
            "base_sample_idx": base_sample_idx,
            "visit_sample_idx": visit_sample_idx,
            "run_seed": int(seed_label),
            "walk_count": walk_count,
            "total_nodes": metadata.get("statistics", {}).get("total_nodes", ""),
            "leaf_nodes": metadata.get("statistics", {}).get("leaf_nodes", ""),
            "max_depth": metadata.get("statistics", {}).get("max_depth", ""),
            "model_path": metadata.get("model_path", self.model_path),
            "generated_at": metadata.get("generated_at", ""),
            "terminal_reason": leaf.get("terminal_reason", ""),
        }

    def _collect_jobs(
        self,
        tree_dir: Path,
        num_samples: int,
        completed_keys: set[Tuple[str, str, int]],
    ) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
        json_files = [
            path
            for path in sorted(tree_dir.glob("*.json"))
            if path.name != "batch_summary.json"
        ]
        eos_jobs: List[Dict[str, Any]] = []
        inference_jobs: List[Dict[str, Any]] = []
        terminator_ids = set(self.terminator_token_ids)

        for json_file in json_files:
            with open(json_file, "r", encoding="utf-8") as handle:
                data = json.load(handle)
            metadata = data.get("metadata", {})
            seed_index_map = _seed_sample_index_map(metadata)
            leaves = collect_walk_leaves(data.get("tree", {}))

            if leaves:
                print(f"  {json_file.name}: {len(leaves)} walked leaves")

            for leaf in leaves:
                for seed_label, count in leaf["walk_counts"].items():
                    if seed_label in seed_index_map:
                        base_sample_idx = seed_index_map[seed_label]
                    else:
                        base_sample_idx = int(seed_label) % max(1, num_samples)

                    is_eos = (
                        bool(leaf["token_ids"])
                        and int(leaf["token_ids"][-1]) in terminator_ids
                    )
                    effective_count = 1 if is_eos else int(count)
                    for visit_sample_idx in range(effective_count):
                        job = self._build_job(
                            file_name=json_file.name,
                            metadata=metadata,
                            leaf=leaf,
                            seed_label=seed_label,
                            base_sample_idx=base_sample_idx,
                            visit_sample_idx=visit_sample_idx,
                            walk_count=int(count),
                            num_samples=num_samples,
                        )
                        key = (job["file_name"], job["leaf_id"], job["sample_idx"])
                        if key in completed_keys:
                            continue
                        if is_eos:
                            eos_jobs.append(job)
                        else:
                            inference_jobs.append(job)

        return eos_jobs, inference_jobs

    def _job_seed(self, job: Dict[str, Any], temperature: float) -> Optional[int]:
        if temperature <= 0:
            return None
        if int(job.get("visit_sample_idx", 0)) == 0:
            return int(job["run_seed"])
        digest = hashlib.sha256(
            (
                f"{job['run_seed']}:{job['file_name']}:{job['leaf_id']}:"
                f"{job['visit_sample_idx']}"
            ).encode("utf-8")
        ).digest()
        return int.from_bytes(digest[:4], "big") & 0x7FFFFFFF

    def _build_row(
        self,
        job: Dict[str, Any],
        continuation: str,
        is_eos: bool,
    ) -> Dict[str, Any]:
        prefix_text = "".join(job["texts"])
        return {
            "file_name": job["file_name"],
            "response": prefix_text,
            "leaf_id": job["leaf_id"],
            "depth": job["depth"],
            "total_nodes": job["total_nodes"],
            "leaf_nodes": job["leaf_nodes"],
            "max_depth": job["max_depth"],
            "model_path": job["model_path"],
            "generated_at": job["generated_at"],
            "path_probability": job["path_probability"],
            "greedy_from_depth": "",
            "sample_idx": job["sample_idx"],
            "base_sample_idx": job["base_sample_idx"],
            "visit_sample_idx": job["visit_sample_idx"],
            "prefix_used": prefix_text,
            "continuation": continuation,
            "full_continuation": prefix_text + continuation,
            "is_eos": is_eos,
            "source": "hybrid_walk",
            "walk_count": job["walk_count"],
            "run_seed": job.get("run_seed", ""),
            "terminal_reason": job.get("terminal_reason", ""),
        }

    def generate(
        self,
        tree_dir: str,
        output_path: str,
        max_tokens: int = 200,
        temperature: float = 0.6,
        top_p: float = 0.9,
        top_k: int = -1,
        batch_size: int = 2000,
        num_samples: int = 1,
        resume: bool = False,
    ) -> pd.DataFrame:
        assert self.llm is not None
        assert self.tokenizer is not None
        assert self.template is not None
        if batch_size < 1:
            raise ValueError("batch_size must be >= 1")
        if num_samples < 1:
            raise ValueError("num_samples must be >= 1")
        if max_tokens < 1:
            raise ValueError("max_tokens must be >= 1")

        tree_dir_path = Path(tree_dir)
        output = Path(output_path)
        output.parent.mkdir(parents=True, exist_ok=True)

        completed_keys: set[Tuple[str, str, int]] = set()
        previous_df = pd.DataFrame()
        if resume and output.exists():
            previous_df = pd.read_csv(output)
            for _, row in previous_df.iterrows():
                completed_keys.add(
                    (
                        str(row["file_name"]),
                        str(row["leaf_id"]),
                        int(row["sample_idx"]),
                    )
                )
            print(f"Resuming: {len(completed_keys)} rows already complete")

        print(f"\n[Phase 1] Collecting hybrid walk jobs from {tree_dir_path}...")
        eos_jobs, inference_jobs = self._collect_jobs(
            tree_dir=tree_dir_path,
            num_samples=max(1, int(num_samples)),
            completed_keys=completed_keys,
        )
        print(f"  EOS jobs: {len(eos_jobs)}")
        print(f"  Inference jobs: {len(inference_jobs)}")

        generated_rows: List[Dict[str, Any]] = []
        for job in eos_jobs:
            generated_rows.append(
                self._build_row(job, continuation="", is_eos=True)
            )

        if inference_jobs:
            print(f"\n[Phase 2] Generating {len(inference_jobs)} continuations...")
            effective_top_k = top_k if top_k > 0 else -1

            for job in inference_jobs:
                base_prompt = self.template["prompt"].format(instruction=job["prompt"])
                base_tokens = self.tokenizer.encode(base_prompt, add_special_tokens=True)
                job["full_token_ids"] = base_tokens + job["token_ids"]

            total_batches = (len(inference_jobs) + batch_size - 1) // batch_size
            for batch_idx in range(total_batches):
                start = batch_idx * batch_size
                end = min(start + batch_size, len(inference_jobs))
                batch = inference_jobs[start:end]

                prompts = [{"prompt_token_ids": job["full_token_ids"]} for job in batch]
                sampling_params = []
                for job in batch:
                    kwargs: Dict[str, Any] = {
                        "max_tokens": max(1, max_tokens - int(job["depth"])),
                        "stop_token_ids": self.terminator_token_ids,
                    }
                    if temperature > 0:
                        kwargs.update({
                            "temperature": temperature,
                            "top_p": top_p,
                            "top_k": effective_top_k,
                        })
                        job_seed = self._job_seed(job, temperature)
                        if job_seed is not None:
                            kwargs["seed"] = job_seed
                    else:
                        kwargs["temperature"] = 0.0
                    sampling_params.append(SamplingParams(**kwargs))

                outputs = self.llm.generate(prompts, sampling_params=sampling_params)
                for job, output_item in zip(batch, outputs):
                    continuation = output_item.outputs[0].text
                    generated_rows.append(
                        self._build_row(job, continuation=continuation, is_eos=False)
                    )

                print(f"  [{batch_idx + 1}/{total_batches}] {end}/{len(inference_jobs)} done")
        else:
            print("\n[Phase 2] No inference jobs")

        new_df = pd.DataFrame(generated_rows)
        if resume and not previous_df.empty:
            result_df = pd.concat([previous_df, new_df], ignore_index=True)
        else:
            result_df = new_df

        if not result_df.empty:
            result_df = result_df.sort_values(
                ["file_name", "leaf_id", "sample_idx"]
            ).reset_index(drop=True)

        with tempfile.NamedTemporaryFile(
            "w",
            dir=output.parent,
            delete=False,
            suffix=".tmp",
        ) as handle:
            result_df.to_csv(handle, index=False)
            temp_path = handle.name
        shutil.move(temp_path, output)

        print(f"\nResults saved to {output}")
        print(f"  Total rows: {len(result_df)}")
        print(f"  Newly generated rows: {len(generated_rows)}")
        return result_df


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(
        description="Generate continuations from hybrid tree walk_counts"
    )
    parser.add_argument("--tree-dir", "-t", required=True, help="Hybrid tree JSON directory")
    parser.add_argument("--output", "-o", required=True, help="Output CSV")
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--model-type", default="llama2")
    parser.add_argument("--dtype", default="float16")
    parser.add_argument("--num-gpus", type=int, default=2)
    parser.add_argument("--max-model-len", type=int, default=2048)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.7)
    parser.add_argument("--max-tokens", type=int, default=200)
    parser.add_argument("--temperature", type=float, default=0.6)
    parser.add_argument("--top-p", type=float, default=0.9)
    parser.add_argument("--top-k", type=int, default=-1)
    parser.add_argument("--batch-size", type=int, default=2000)
    parser.add_argument("--num-samples", type=int, default=1)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--system-prompt-suffix", type=str, default=None)
    args = parser.parse_args()

    if args.system_prompt_suffix:
        with open(args.system_prompt_suffix, "r", encoding="utf-8") as handle:
            set_system_prompt_suffix(handle.read().strip())

    generator = HybridWalkContinuationGenerator(
        model_path=args.model_path,
        model_type=args.model_type,
        dtype=args.dtype,
        num_gpus=args.num_gpus,
        max_model_len=args.max_model_len,
        gpu_memory_utilization=args.gpu_memory_utilization,
    )
    generator.initialize()
    try:
        generator.generate(
            tree_dir=args.tree_dir,
            output_path=args.output,
            max_tokens=args.max_tokens,
            temperature=args.temperature,
            top_p=args.top_p,
            top_k=args.top_k,
            batch_size=args.batch_size,
            num_samples=args.num_samples,
            resume=args.resume,
        )
    finally:
        generator.cleanup()


if __name__ == "__main__":
    main()
