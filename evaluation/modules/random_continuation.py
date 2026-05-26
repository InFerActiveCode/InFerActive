"""Random continuation via tree walk simulation.

Takes an existing continued.csv and:
1. Simulates M random walks through each tree and seed sample (weighted by stored probs)
2. Reuses existing continuations for visited leaves
3. Generates additional continuations only where visit_count > 1
4. Outputs continued_with_random.csv with a 'source' column to distinguish

This avoids redundant inference — only (visit_count - 1) extra samples
per leaf are generated.
"""

import os
import sys
import json
import random
import tempfile
import shutil
from collections import Counter
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd
from vllm import LLM, SamplingParams
from transformers import AutoTokenizer

if __name__ == "__main__":
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

try:
    from .model_utils import get_template
except ImportError:
    from model_utils import get_template


def _get_terminator_token_ids(tokenizer: AutoTokenizer, model_type: str) -> List[int]:
    """Return model terminator token ids."""
    terminator_ids: List[int] = []
    eos_token_id = getattr(tokenizer, "eos_token_id", None)

    if isinstance(eos_token_id, int):
        terminator_ids.append(eos_token_id)
    elif isinstance(eos_token_id, (list, tuple)):
        terminator_ids.extend(int(tid) for tid in eos_token_id)

    if model_type == "llama3":
        terminator_ids.extend([128001, 128008, 128009])
    if model_type == "qwen":
        endoftext = tokenizer.convert_tokens_to_ids("<|endoftext|>")
        if isinstance(endoftext, int) and endoftext >= 0:
            terminator_ids.append(endoftext)
    if model_type == "gemma":
        eot = tokenizer.convert_tokens_to_ids("<end_of_turn>")
        if isinstance(eot, int) and eot >= 0:
            terminator_ids.append(eot)

    return list(dict.fromkeys(terminator_ids))


# ──────────────────────────────────────────────────────
#  Tree walk simulation
# ──────────────────────────────────────────────────────

def _walk_tree_once(tree: Dict) -> Tuple[str, List[int], List[str], float]:
    """One random walk from root to leaf.

    Returns (leaf_id, token_ids, texts, path_probability).
    """
    node = tree
    token_ids: List[int] = []
    texts: List[str] = []
    path_prob = 1.0

    while True:
        children = node.get("children", [])
        if not children:
            break

        probs = [c.get("prob", 0.0) for c in children]
        total = sum(probs)
        if total <= 0:
            break

        weights = [p / total for p in probs]
        chosen = random.choices(children, weights=weights, k=1)[0]

        token_ids.append(chosen.get("token_id", -1))
        texts.append(chosen.get("text", ""))
        path_prob *= chosen.get("prob", 1.0)
        node = chosen

    return node.get("id", ""), token_ids, texts, path_prob


def simulate_walks(
    tree_data: Dict,
    num_walks: int,
    seed: Optional[int] = None,
) -> Dict[str, Dict]:
    """Simulate num_walks random walks through tree.

    Returns {leaf_id: {"count": N, "token_ids": [...], "texts": [...],
                        "avg_prob": float}}.
    """
    if seed is not None:
        random.seed(seed)

    tree = tree_data["tree"]
    visits: Counter = Counter()
    leaf_info: Dict[str, Dict] = {}

    for _ in range(num_walks):
        leaf_id, token_ids, texts, path_prob = _walk_tree_once(tree)
        visits[leaf_id] += 1
        if leaf_id not in leaf_info:
            leaf_info[leaf_id] = {
                "token_ids": token_ids,
                "texts": texts,
                "path_prob_sum": path_prob,
            }
        else:
            leaf_info[leaf_id]["path_prob_sum"] += path_prob

    result: Dict[str, Dict] = {}
    for lid, count in visits.items():
        info = leaf_info[lid]
        result[lid] = {
            "count": count,
            "token_ids": info["token_ids"],
            "texts": info["texts"],
            "avg_prob": info["path_prob_sum"] / count,
        }
    return result


# ──────────────────────────────────────────────────────
#  Main generator
# ──────────────────────────────────────────────────────

class RandomContinuationGenerator:
    """Generate random-walk continuations reusing existing continued.csv."""

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

    def initialize(self):
        print(f"Loading model from {self.model_path}...")
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_path, use_fast=True, trust_remote_code=True
        )
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self.terminator_token_ids = _get_terminator_token_ids(
            self.tokenizer, self.model_type
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
        print("Model loaded!")

    def cleanup(self):
        print("Cleaning up...")
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

    def generate(
        self,
        tree_dir: str,
        continued_csv: str,
        output_path: str,
        num_walks: int = 10,
        max_tokens: int = 200,
        temperature: float = 0.6,
        top_p: float = 0.9,
        top_k: int = -1,
        seed: int = 42,
        batch_size: int = 2000,
        num_samples: int = 1,
    ) -> pd.DataFrame:
        """Run tree walk + reuse existing + generate additional.

        Args:
            tree_dir: Directory with tree JSON files.
            continued_csv: Path to existing continued.csv.
            output_path: Output CSV path (continued_with_random.csv).
            num_walks: Random walks per tree.
            max_tokens: Max tokens for continuation.
            temperature: Sampling temperature.
            top_p: Top-p sampling.
            top_k: Top-k sampling parameter (-1 disables top-k filtering).
            seed: Random seed. With num_samples > 1, runs use seed, seed+1, ...
            batch_size: vLLM batch size.
            num_samples: Number of independent seed samples.

        Returns:
            DataFrame with combined results.
        """
        num_samples = max(1, int(num_samples))
        tree_dir = Path(tree_dir)
        json_files = sorted(tree_dir.glob("*.json"))
        json_files = [f for f in json_files if f.name != "batch_summary.json"]

        if not json_files:
            print(f"No tree JSON files in {tree_dir}")
            return pd.DataFrame()

        # ── Load existing continued.csv (optional) ──
        if os.path.exists(continued_csv):
            existing_df = pd.read_csv(continued_csv)
            print(f"Loaded {len(existing_df)} rows from {continued_csv}")
        else:
            print(f"No continued.csv found at {continued_csv} — generating all continuations from scratch")
            existing_df = pd.DataFrame()

        # Build lookup: (file_name, leaf_id, sample_idx) → row
        existing_lookup: Dict[Tuple[str, str, int], pd.Series] = {}
        for _, row in existing_df.iterrows():
            sample_idx = (
                int(row["sample_idx"])
                if "sample_idx" in row and pd.notna(row["sample_idx"])
                else 0
            )
            key = (str(row["file_name"]), str(row["leaf_id"]), sample_idx)
            existing_lookup[key] = row

        # ── Phase 1: Walk simulation ──
        print(f"\n[Phase 1] Simulating {num_walks} walks per tree x {num_samples} seed samples...")
        walk_plan: Dict[Tuple[str, int], Dict[str, Dict]] = {}
        tree_metadata: Dict[str, Dict] = {}

        for json_file in json_files:
            with open(json_file, "r", encoding="utf-8") as f:
                data = json.load(f)
            tree_metadata[json_file.name] = data.get("metadata", {})

            for base_sample_idx in range(num_samples):
                run_seed = seed + base_sample_idx
                # Per-tree deterministic seed so walks are independent across trees
                import hashlib
                tree_seed_bytes = hashlib.sha256(
                    f"{run_seed}:tree:{json_file.name}".encode()
                ).digest()
                tree_seed = int.from_bytes(tree_seed_bytes[:4], "big")
                walks = simulate_walks(data, num_walks, seed=tree_seed)
                if walks:
                    walk_plan[(json_file.name, base_sample_idx)] = walks

        # Stats
        total_visits = sum(
            info["count"]
            for leaves in walk_plan.values()
            for info in leaves.values()
        )
        reusable = 0
        extra_needed = 0
        for (file_name, base_sample_idx), leaves in walk_plan.items():
            for leaf_id, info in leaves.items():
                count = info["count"]
                key = (file_name, leaf_id, base_sample_idx)
                if key in existing_lookup:
                    reusable += min(1, count)
                    extra_needed += max(0, count - 1)
                else:
                    extra_needed += count

        print(f"  Total walk visits: {total_visits}")
        print(f"  Reusing from continued.csv: {reusable}")
        print(f"  Additional inference needed: {extra_needed}")

        # ── Check resume ──
        reused_rows: List[Dict] = []
        extra_jobs: List[Dict] = []

        completed_keys: set = set()
        if os.path.exists(output_path):
            out_df = pd.read_csv(output_path)
            if "file_name" in out_df.columns and "leaf_id" in out_df.columns and "sample_idx" in out_df.columns:
                for _, r in out_df.iterrows():
                    completed_keys.add((str(r["file_name"]), str(r["leaf_id"]), int(r["sample_idx"])))
                if completed_keys:
                    print(f"  Resuming: {len(completed_keys)} samples already done, skipping")

        # ── Phase 2: Build output rows ──
        # Include ALL leaves from continued.csv.
        # - walk_count > 0: visited by walk, reuse first + generate extras
        # - walk_count = 0: not visited, still included for comparison
        print(f"\n[Phase 2] Building output...")

        # Track which existing leaves are covered by walks
        walked_keys = set()
        for (file_name, base_sample_idx), leaves in walk_plan.items():
            for leaf_id in leaves:
                walked_keys.add((file_name, leaf_id, base_sample_idx))

        # 2a. Walk-visited leaves
        terminator_ids = set(self.terminator_token_ids)
        for (file_name, base_sample_idx), leaves in walk_plan.items():
            meta = tree_metadata.get(file_name, {})
            prompt = meta.get("prompt", "")
            run_seed = seed + base_sample_idx

            for leaf_id, info in leaves.items():
                count = info["count"]
                key = (file_name, leaf_id, base_sample_idx)
                existing_row = existing_lookup.get(key)

                # EOS leaves: cap to 1 sample (duplicates are identical)
                is_eos_leaf = info["token_ids"] and info["token_ids"][-1] in terminator_ids
                effective_count = 1 if is_eos_leaf else count

                # First sample: reuse existing if available
                if existing_row is not None:
                    resume_key = (file_name, leaf_id, base_sample_idx)
                    if resume_key not in completed_keys:
                        row_dict = existing_row.to_dict()
                        row_dict["source"] = "continued"
                        row_dict["walk_count"] = count
                        row_dict["sample_idx"] = base_sample_idx
                        row_dict["run_seed"] = run_seed
                        reused_rows.append(row_dict)

                    row_dict = existing_row.to_dict()
                    # Additional samples needed
                    for visit_sample_idx in range(1, effective_count):
                        output_sample_idx = base_sample_idx + visit_sample_idx * num_samples
                        if (file_name, leaf_id, output_sample_idx) in completed_keys:
                            continue
                        extra_jobs.append({
                            "file_name": file_name,
                            "leaf_id": leaf_id,
                            "prompt": prompt,
                            "token_ids": info["token_ids"],
                            "texts": info["texts"],
                            "depth": len(info["texts"]),
                            "avg_prob": info["avg_prob"],
                            "sample_idx": output_sample_idx,
                            "base_sample_idx": base_sample_idx,
                            "visit_sample_idx": visit_sample_idx,
                            "run_seed": run_seed,
                            "walk_count": count,
                            "existing_row": row_dict,
                        })
                else:
                    # No existing continuation — generate all
                    for visit_sample_idx in range(effective_count):
                        output_sample_idx = base_sample_idx + visit_sample_idx * num_samples
                        if (file_name, leaf_id, output_sample_idx) in completed_keys:
                            continue
                        extra_jobs.append({
                            "file_name": file_name,
                            "leaf_id": leaf_id,
                            "prompt": prompt,
                            "token_ids": info["token_ids"],
                            "texts": info["texts"],
                            "depth": len(info["texts"]),
                            "avg_prob": info["avg_prob"],
                            "sample_idx": output_sample_idx,
                            "base_sample_idx": base_sample_idx,
                            "visit_sample_idx": visit_sample_idx,
                            "run_seed": run_seed,
                            "walk_count": count,
                            "existing_row": None,
                        })

        # 2b. Non-visited leaves — include from continued.csv with walk_count=0
        for key, row in existing_lookup.items():
            file_name, leaf_id, base_sample_idx = key
            if key not in walked_keys and key not in completed_keys:
                row_dict = row.to_dict()
                row_dict["source"] = "continued"
                row_dict["walk_count"] = 0
                row_dict["sample_idx"] = base_sample_idx
                row_dict["run_seed"] = seed + base_sample_idx
                reused_rows.append(row_dict)

        print(f"  Reused rows: {len(reused_rows)} (walk_count=0: {sum(1 for r in reused_rows if r['walk_count'] == 0)})")
        print(f"  Extra jobs: {len(extra_jobs)}")

        # ── Phase 3: Generate additional continuations ──
        generated_rows: List[Dict] = []

        if extra_jobs:
            # Separate EOS vs inference jobs
            terminator_ids = set(self.terminator_token_ids)
            eos_jobs = [
                j for j in extra_jobs
                if j["token_ids"] and j["token_ids"][-1] in terminator_ids
            ]
            inference_jobs = [
                j for j in extra_jobs
                if not j["token_ids"] or j["token_ids"][-1] not in terminator_ids
            ]

            # EOS jobs — no inference needed
            for j in eos_jobs:
                prefix_text = "".join(j["texts"])
                generated_rows.append(
                    self._build_row(j, continuation="", prefix_text=prefix_text, is_eos=True)
                )

            # Inference jobs
            if inference_jobs:
                print(f"\n[Phase 3] Generating {len(inference_jobs)} additional continuations...")
                effective_top_k = top_k if top_k > 0 else -1
                if temperature > 0:
                    sampling_desc = (
                        f"temperature={temperature}, top_p={top_p}, top_k={effective_top_k}"
                    )
                    if seed is not None:
                        sampling_desc = f"{sampling_desc}, seed={seed}"
                    print(f"  Using random sampling: {sampling_desc}")
                else:
                    print("  Using greedy decoding (temperature=0.0)")

                # Build token inputs
                for j in inference_jobs:
                    base_prompt = self.template["prompt"].format(instruction=j["prompt"])
                    base_tokens = self.tokenizer.encode(base_prompt, add_special_tokens=True)
                    j["full_token_ids"] = base_tokens + j["token_ids"]

                total_batches = (len(inference_jobs) + batch_size - 1) // batch_size

                for batch_idx in range(total_batches):
                    start = batch_idx * batch_size
                    end = min(start + batch_size, len(inference_jobs))
                    batch = inference_jobs[start:end]

                    prompts = [
                        {"prompt_token_ids": j["full_token_ids"]} for j in batch
                    ]
                    sp_list = []
                    for idx, j in enumerate(batch):
                        # The first generated row for a seed sample uses the
                        # seed directly. Extra rows for repeated visits derive
                        # deterministic seeds to avoid duplicate outputs.
                        if temperature > 0 and seed is not None:
                            if j.get("visit_sample_idx", 0) == 0:
                                job_seed = int(j["run_seed"])
                            else:
                                import hashlib
                                h = hashlib.sha256(
                                    f"{j['run_seed']}:{j['file_name']}:{j['leaf_id']}:{j['visit_sample_idx']}".encode()
                                ).digest()
                                job_seed = int.from_bytes(h[:4], "big") & 0x7FFFFFFF
                        else:
                            job_seed = None

                        sampling_kwargs = {
                            "max_tokens": max(1, max_tokens - j["depth"]),
                            "stop_token_ids": self.terminator_token_ids,
                        }
                        if temperature > 0:
                            sampling_kwargs.update({
                                "temperature": temperature,
                                "top_p": top_p,
                                "top_k": effective_top_k,
                            })
                            if job_seed is not None:
                                sampling_kwargs["seed"] = job_seed
                        else:
                            sampling_kwargs["temperature"] = 0.0

                        sp_list.append(SamplingParams(**sampling_kwargs))

                    outputs = self.llm.generate(prompts, sampling_params=sp_list)

                    for j, o in zip(batch, outputs):
                        continuation = o.outputs[0].text
                        prefix_text = "".join(j["texts"])
                        generated_rows.append(
                            self._build_row(j, continuation=continuation, prefix_text=prefix_text, is_eos=False)
                        )

                    print(f"  [{batch_idx+1}/{total_batches}] {end}/{len(inference_jobs)} done")
        else:
            print("\n[Phase 3] No additional inference needed")

        # ── Phase 4: Combine and save ──
        print(f"\n[Phase 4] Combining results...")

        new_df = pd.DataFrame(reused_rows + generated_rows)

        # Append to existing output if resuming
        if completed_keys and os.path.exists(output_path):
            prev_df = pd.read_csv(output_path)
            result_df = pd.concat([prev_df, new_df], ignore_index=True)
        else:
            result_df = new_df

        # Ensure consistent columns — keep all columns from existing + source/walk_count/sample_idx
        result_df = result_df.sort_values(
            ["file_name", "leaf_id", "sample_idx"]
        ).reset_index(drop=True)

        # Atomic write
        with tempfile.NamedTemporaryFile(
            "w", dir=Path(output_path).parent, delete=False, suffix=".tmp"
        ) as tmp:
            result_df.to_csv(tmp, index=False)
            tmp_path = tmp.name
        shutil.move(tmp_path, output_path)

        print(f"\nResults saved to {output_path}")
        print(f"  Total rows: {len(result_df)}")
        print(f"  From continued.csv: {len(reused_rows)}")
        print(f"  Newly generated: {len(generated_rows)}")

        return result_df

    def _build_row(self, job: Dict, continuation: str, prefix_text: str, is_eos: bool) -> Dict:
        """Build an output row from a job dict."""
        existing = job.get("existing_row")

        if existing:
            # Copy columns from existing row for consistency
            row = {
                "file_name": job["file_name"],
                "response": existing.get("response", prefix_text),
                "leaf_id": job["leaf_id"],
                "depth": job["depth"],
                "total_nodes": existing.get("total_nodes", ""),
                "leaf_nodes": existing.get("leaf_nodes", ""),
                "max_depth": existing.get("max_depth", ""),
                "model_path": existing.get("model_path", ""),
                "generated_at": existing.get("generated_at", ""),
                "path_probability": job["avg_prob"],
                "greedy_from_depth": existing.get("greedy_from_depth", ""),
                "sample_idx": job["sample_idx"],
                "prefix_used": prefix_text,
                "continuation": continuation,
                "full_continuation": prefix_text + continuation,
                "is_eos": is_eos,
                "source": "walk_extra",
                "walk_count": job["walk_count"],
                "run_seed": job.get("run_seed", ""),
            }
        else:
            row = {
                "file_name": job["file_name"],
                "response": prefix_text,
                "leaf_id": job["leaf_id"],
                "depth": job["depth"],
                "total_nodes": "",
                "leaf_nodes": "",
                "max_depth": "",
                "model_path": "",
                "generated_at": "",
                "path_probability": job["avg_prob"],
                "greedy_from_depth": "",
                "sample_idx": job["sample_idx"],
                "prefix_used": prefix_text,
                "continuation": continuation,
                "full_continuation": prefix_text + continuation,
                "is_eos": is_eos,
                "source": "walk_extra",
                "walk_count": job["walk_count"],
                "run_seed": job.get("run_seed", ""),
            }
        return row


def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="Random continuation via tree walk + existing continued.csv"
    )
    parser.add_argument("--tree-dir", "-t", required=True, help="Tree JSON directory")
    parser.add_argument("--continued-csv", "-i", required=True,
                        help="Existing continued.csv")
    parser.add_argument("--output", "-o", required=True,
                        help="Output CSV (continued_with_random.csv)")
    parser.add_argument("--model-path", required=True, help="Path to model")
    parser.add_argument("--model-type", default="llama2")
    parser.add_argument("--dtype", default="float16")
    parser.add_argument("--num-gpus", type=int, default=2)
    parser.add_argument("--max-model-len", type=int, default=2048)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.7)
    parser.add_argument("--num-walks", "-n", type=int, default=10)
    parser.add_argument("--max-tokens", type=int, default=200)
    parser.add_argument("--temperature", type=float, default=0.6)
    parser.add_argument("--top-p", type=float, default=0.9)
    parser.add_argument("--top-k", type=int, default=-1,
                        help="Top-k sampling parameter (-1 disables top-k filtering)")
    parser.add_argument("--seed", "-s", type=int, default=42)
    parser.add_argument("--batch-size", type=int, default=2000)
    parser.add_argument("--num-samples", type=int, default=1,
                        help="Number of independent seed samples (seed, seed+1, ...)")
    parser.add_argument("--system-prompt-suffix", type=str, default=None)
    args = parser.parse_args()

    if args.system_prompt_suffix:
        from model_utils import set_system_prompt_suffix
        with open(args.system_prompt_suffix, "r", encoding="utf-8") as f:
            set_system_prompt_suffix(f.read().strip())

    gen = RandomContinuationGenerator(
        model_path=args.model_path,
        model_type=args.model_type,
        dtype=args.dtype,
        num_gpus=args.num_gpus,
        max_model_len=args.max_model_len,
        gpu_memory_utilization=args.gpu_memory_utilization,
    )
    gen.initialize()

    gen.generate(
        tree_dir=args.tree_dir,
        continued_csv=args.continued_csv,
        output_path=args.output,
        num_walks=args.num_walks,
        max_tokens=args.max_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
        top_k=args.top_k,
        seed=args.seed,
        batch_size=args.batch_size,
        num_samples=args.num_samples,
    )

    gen.cleanup()


if __name__ == "__main__":
    main()
