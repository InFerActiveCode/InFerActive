#!/usr/bin/env python3
"""
Random continuation experiment runner.

Pipeline:
  1. Tree exploration (reuse existing or generate new)
  2. Continuation — N samples per leaf (reuse existing or generate)
  3. Random walk simulation + additional continuations for visited leaves
  4. Classifier on continued_with_random.csv

This reuses the existing run_experiment pipeline for steps 1-2, then adds
the walk + extra continuation step on top.

Usage:
    # Full pipeline from scratch
    python run_random_continuation.py -m llama3_1b -b /path/to/behaviors.csv -n 10

    # Reuse existing trees + continued.csv
    python run_random_continuation.py -m llama3_1b -b /path/to/behaviors.csv \
        --skip-tree --tree-dir results/llama3_1b/trees \
        --skip-continuation --continued-csv results/llama3_1b/continued.csv \
        -n 10
"""

import os
import sys
import argparse
import json
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Optional

import yaml


DEFAULT_CUDA_VISIBLE_DEVICES = "2,3,4,5"
DEFAULT_CATEGORIES = ["contextual", "standard"]
CONFIG_SECTION_KEYS = {"classifier", "generation", "categories"}
STEP_SEPARATOR = "=" * 60


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run random continuation experiment (tree + continuation + walk)"
    )

    # Required
    parser.add_argument("--model-name", "-m", required=True,
                        help="Model name from config/models.yaml")
    parser.add_argument("--behaviors-csv", "-b", required=True,
                        help="Path to HarmBench behaviors CSV")

    # Output
    parser.add_argument("--output-dir", "-o", default="./results")
    parser.add_argument("--output-name", type=str, default=None,
                        help="Output subdirectory name (default: model name)")
    parser.add_argument("--config-file", "-c", default="./config/models.yaml")

    # Behavior selection
    parser.add_argument("--categories", nargs="+", default=None)
    parser.add_argument("--start-idx", type=int, default=0)
    parser.add_argument("--max-count", type=int, default=None)

    # Walk parameters
    parser.add_argument("--num-walks", "-n", type=int, default=10,
                        help="Number of random walks per tree")

    # Tree parameters
    parser.add_argument("--exploration-depths", nargs="+", type=int, default=None)
    parser.add_argument("--k", type=int, default=None)
    parser.add_argument("--temperature", type=float, default=None)
    parser.add_argument("--top-p", type=float, default=None)

    # Continuation parameters
    parser.add_argument("--max-tokens", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--top-k", type=int, default=None,
                        help="Top-k sampling parameter for continuation steps (-1 disables top-k filtering)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-samples", type=int, default=1,
                        help="Number of independent seed samples (seed, seed+1, ...)")

    # Pipeline control
    parser.add_argument("--skip-tree", action="store_true",
                        help="Skip tree exploration")
    parser.add_argument("--tree-dir", type=str, default=None,
                        help="Path to existing tree directory")
    parser.add_argument("--skip-continuation", action="store_true",
                        help="Skip base continuation")
    parser.add_argument("--continued-csv", type=str, default=None,
                        help="Path to existing continued.csv")
    parser.add_argument("--skip-walk", action="store_true",
                        help="Skip walk + extra continuation step")
    parser.add_argument("--skip-classifier", action="store_true")
    parser.add_argument("--resume", action="store_true")

    # GPU
    parser.add_argument("--gpu-ids", type=str, default=None)

    # System prompt
    parser.add_argument("--system-prompt-suffix", type=str, default=None)

    return parser.parse_args()


def load_config(config_path: str) -> Dict:
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def set_cuda_visible_devices(gpu_ids: Optional[str]) -> None:
    if gpu_ids:
        os.environ["CUDA_VISIBLE_DEVICES"] = gpu_ids
    elif "CUDA_VISIBLE_DEVICES" not in os.environ:
        os.environ["CUDA_VISIBLE_DEVICES"] = DEFAULT_CUDA_VISIBLE_DEVICES


def available_model_names(config: Dict) -> List[str]:
    return [key for key in config if key not in CONFIG_SECTION_KEYS]


def config_value(args, gen_config: Dict, name: str, default):
    value = getattr(args, name, None)
    if value is not None:
        return value
    return gen_config.get(name, default)


def run_command(cmd: list, description: str) -> int:
    import subprocess

    print(f"\n{STEP_SEPARATOR}")
    print(f"[STEP] {description}")
    print(STEP_SEPARATOR)
    print(f"Command: {' '.join(cmd)}\n", flush=True)

    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"
    result = subprocess.run(cmd, env=env)
    return result.returncode


def run_tree_step(model_config, gen_config, tree_dir, args, categories) -> bool:
    """Run tree exploration subprocess."""
    script_dir = Path(__file__).parent

    exploration_depths = args.exploration_depths if args.exploration_depths is not None else gen_config.get(
        "exploration_depths", [3, 3, 3, 3, 3, 3, 2]
    )
    k = config_value(args, gen_config, "k", 5)
    temperature = config_value(args, gen_config, "temperature", 0.7)
    top_p = config_value(args, gen_config, "top_p", 0.9)
    min_p = gen_config.get("min_p", 0.0)
    tree_batch_size = gen_config.get("tree_batch_size", 128)

    cmd = [
        sys.executable, str(script_dir / "modules" / "tree_explorer_vllm.py"),
        "--behaviors-csv", args.behaviors_csv,
        "--model-path", model_config["model_path"],
        "--model-type", model_config["model_type"],
        "--dtype", str(model_config.get("dtype", "float16")),
        "--output-dir", str(tree_dir),
        "--exploration-depths", *[str(d) for d in exploration_depths],
        "--k", str(k),
        "--temperature", str(temperature),
        "--top-p", str(top_p),
        "--min-p", str(min_p),
        "--batch-size", str(tree_batch_size),
        "--num-gpus", str(model_config.get("num_gpus", 2)),
        "--max-model-len", str(model_config.get("max_model_len", 2048)),
        "--gpu-memory-utilization", str(model_config.get("gpu_memory_utilization", 0.7)),
        "--start-idx", str(args.start_idx),
        "--categories", *categories,
    ]
    if args.max_count:
        cmd.extend(["--max-count", str(args.max_count)])
    if args.system_prompt_suffix:
        cmd.extend(["--system-prompt-suffix", args.system_prompt_suffix])

    return run_command(cmd, "Tree Exploration") == 0


def run_continuation_step(model_config, gen_config, tree_dir, output_path, args, categories) -> bool:
    """Run base continuation subprocess."""
    script_dir = Path(__file__).parent

    max_tokens = config_value(args, gen_config, "max_tokens", 200)
    batch_size = config_value(args, gen_config, "batch_size", 2000)
    temperature = config_value(args, gen_config, "temperature", 0.6)
    top_p = config_value(args, gen_config, "top_p", 0.9)
    top_k = args.top_k if args.top_k is not None else gen_config.get("top_k", gen_config.get("k", -1))

    cmd = [
        sys.executable, str(script_dir / "modules" / "continuation.py"),
        "--input", str(tree_dir),
        "--output", output_path,
        "--model-path", model_config["model_path"],
        "--model-type", model_config["model_type"],
        "--dtype", str(model_config.get("dtype", "float16")),
        "--num-gpus", str(model_config.get("num_gpus", 2)),
        "--max-model-len", str(model_config.get("max_model_len", 2048)),
        "--gpu-memory-utilization", str(model_config.get("gpu_memory_utilization", 0.7)),
        "--batch-size", str(batch_size),
        "--max-tokens", str(max_tokens),
        "--temperature", str(temperature),
        "--top-p", str(top_p),
        "--top-k", str(top_k),
        "--num-samples", str(args.num_samples),
    ]
    if args.seed is not None:
        cmd.extend(["--seed", str(args.seed)])
    if args.system_prompt_suffix:
        cmd.extend(["--system-prompt-suffix", args.system_prompt_suffix])
    if args.resume:
        cmd.append("--resume")

    return run_command(cmd, f"Base Continuation ({args.num_samples} per leaf)") == 0


def run_walk_continuation_step(model_config, gen_config, tree_dir, continued_csv, output_path, args) -> bool:
    """Run walk simulation + additional continuations subprocess."""
    script_dir = Path(__file__).parent

    max_tokens = config_value(args, gen_config, "max_tokens", 200)
    batch_size = config_value(args, gen_config, "batch_size", 2000)
    temperature = config_value(args, gen_config, "temperature", 0.6)
    top_p = config_value(args, gen_config, "top_p", 0.9)
    top_k = args.top_k if args.top_k is not None else gen_config.get("top_k", gen_config.get("k", -1))

    cmd = [
        sys.executable, str(script_dir / "modules" / "random_continuation.py"),
        "--tree-dir", str(tree_dir),
        "--continued-csv", continued_csv,
        "--output", output_path,
        "--model-path", model_config["model_path"],
        "--model-type", model_config["model_type"],
        "--dtype", str(model_config.get("dtype", "float16")),
        "--num-gpus", str(model_config.get("num_gpus", 2)),
        "--max-model-len", str(model_config.get("max_model_len", 2048)),
        "--gpu-memory-utilization", str(model_config.get("gpu_memory_utilization", 0.7)),
        "--num-walks", str(args.num_walks),
        "--max-tokens", str(max_tokens),
        "--temperature", str(temperature),
        "--top-p", str(top_p),
        "--top-k", str(top_k),
        "--seed", str(args.seed),
        "--batch-size", str(batch_size),
        "--num-samples", str(args.num_samples),
    ]
    if args.system_prompt_suffix:
        cmd.extend(["--system-prompt-suffix", args.system_prompt_suffix])

    return run_command(cmd, f"Walk ({args.num_walks}) + Extra Continuations") == 0


def run_classifier_step(input_path, output_path, classifier_config, args, categories):
    """Run classifier subprocess."""
    script_dir = Path(__file__).parent

    safe_candidate = str(Path(input_path).with_suffix(".classifier_safe.csv"))
    if Path(safe_candidate).exists():
        print(f"[classifier_preprocess] Reusing existing safe file: {safe_candidate}")
        safe_input_path = safe_candidate
    else:
        try:
            from modules.classifier_preprocess import preprocess_experiment_classifier_inputs
            safe_input_path, _ = preprocess_experiment_classifier_inputs(
                continued_input_path=input_path,
                continued_output_path=safe_candidate,
                greedy_input_path=None,
                greedy_output_path=None,
                behaviors_csv=args.behaviors_csv,
                classifier_model_path=classifier_config["model_path"],
                categories=categories,
                start_idx=args.start_idx,
                max_count=args.max_count,
            )
        except Exception as e:
            print(f"[WARN] Preprocessing failed ({e}), using raw input")
            safe_input_path = input_path

    cmd = [
        sys.executable, str(script_dir / "modules" / "classifier.py"),
        "--input", safe_input_path,
        "--output", output_path,
        "--model-path", classifier_config["model_path"],
        "--num-gpus", str(classifier_config.get("num_gpus", 2)),
        "--max-model-len", str(classifier_config.get("max_model_len", 2048)),
        "--gpu-memory-utilization", str(classifier_config.get("gpu_memory_utilization", 0.7)),
        "--behaviors-csv", args.behaviors_csv,
        "--generation-col", "full_continuation",
        "--batch-size", str(args.batch_size if args.batch_size is not None else 2000),
    ]
    if args.resume:
        cmd.append("--resume")

    run_command(cmd, "Classification")


def generate_summary(output_dir, args, config, categories):
    import pandas as pd

    summary = {
        "experiment": {
            "type": "random_continuation",
            "model_name": args.model_name,
            "timestamp": datetime.now().isoformat(),
            "categories": categories,
            "num_walks": args.num_walks,
            "num_samples": args.num_samples,
        },
        "config": {
            "model": config.get(args.model_name, {}),
            "generation": config.get("generation", {}),
        },
        "results": {},
    }

    classified_path = output_dir / "classified.csv"
    if classified_path.exists():
        df = pd.read_csv(classified_path, low_memory=False)
        yes_count = sum(1 for r in df["classifier_result"] if str(r).lower() == "yes")
        total = len(df)

        if "file_name" in df.columns:
            prompt_success = df.groupby("file_name")["classifier_result"].apply(
                lambda x: any(str(r).lower() == "yes" for r in x)
            )
            prompts_with_success = int(prompt_success.sum())
            n_prompts = int(df["file_name"].nunique())
        else:
            prompts_with_success = "N/A"
            n_prompts = "N/A"

        # Source breakdown
        source_counts = {}
        if "source" in df.columns:
            source_counts = df["source"].value_counts().to_dict()

        summary["results"] = {
            "total_samples": total,
            "n_prompts": n_prompts,
            "yes_count": yes_count,
            "no_count": total - yes_count,
            "sample_success_rate": yes_count / total if total > 0 else 0.0,
            "prompts_with_at_least_one_yes": prompts_with_success,
            "prompt_success_rate": (
                prompts_with_success / n_prompts
                if isinstance(n_prompts, int) and n_prompts > 0
                else "N/A"
            ),
            "source_breakdown": {str(k): int(v) for k, v in source_counts.items()},
        }

    summary_path = output_dir / "summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\nSummary saved to {summary_path}")
    return summary


def main():
    args = parse_args()

    set_cuda_visible_devices(args.gpu_ids)

    config = load_config(args.config_file)

    if args.model_name not in config:
        print(f"Error: Model '{args.model_name}' not found in config")
        print(f"Available: {available_model_names(config)}")
        sys.exit(1)

    model_config = config[args.model_name]
    classifier_config = config.get("classifier", {})
    gen_config = config.get("generation", {})
    categories = args.categories or config.get("categories", DEFAULT_CATEGORIES)

    dir_name = args.output_name or args.model_name
    output_dir = Path(args.output_dir) / dir_name / "random_continuation"
    output_dir.mkdir(parents=True, exist_ok=True)

    tree_dir = Path(args.tree_dir) if args.tree_dir else output_dir / "trees"
    tree_dir.mkdir(parents=True, exist_ok=True)

    continued_csv = args.continued_csv or str(output_dir / "continued.csv")
    walk_output = str(output_dir / "continued_with_random.csv")

    print(f"Model: {args.model_name}")
    print(f"Output: {output_dir}")
    print(f"Trees: {tree_dir}")
    print(f"Continued CSV: {continued_csv}")
    print(f"Walks: {args.num_walks}")
    print(f"Samples: {args.num_samples}")
    print(f"Categories: {categories}")

    # ── Step 1: Tree ──
    if not args.skip_tree:
        if not run_tree_step(model_config, gen_config, tree_dir, args, categories):
            print("\n[ABORT] Tree exploration failed")
            sys.exit(1)
    else:
        print("\n[SKIP] Tree exploration")

    # ── Step 2: Base continuation ──
    if not args.skip_continuation:
        if not run_continuation_step(
            model_config, gen_config, tree_dir, continued_csv, args, categories
        ):
            print("\n[ABORT] Base continuation failed")
            sys.exit(1)
    else:
        print("\n[SKIP] Base continuation")

    # ── Step 3: Walk + extra continuations ──
    if not args.skip_walk:
        if not run_walk_continuation_step(
            model_config, gen_config, tree_dir, continued_csv, walk_output, args
        ):
            print("\n[ABORT] Walk + extra continuation failed")
            sys.exit(1)
    else:
        print("\n[SKIP] Walk + extra continuation")

    # ── Step 4: Classifier ──
    # continued_with_random.csv contains ALL leaves:
    #   walk_count=0 → tree-only (not visited by walk)
    #   walk_count>0 → walk-visited (with extra samples)
    # One classifier run covers both; filter by walk_count downstream.
    if not args.skip_classifier and os.path.exists(walk_output):
        classified_path = str(output_dir / "classified.csv")
        run_classifier_step(
            input_path=walk_output,
            output_path=classified_path,
            classifier_config=classifier_config,
            args=args,
            categories=categories,
        )

    # ── Summary ──
    summary = generate_summary(output_dir, args, config, categories)

    print("\n" + STEP_SEPARATOR)
    print("EXPERIMENT COMPLETE")
    print(STEP_SEPARATOR)
    print(f"Results saved to: {output_dir}")

    if "total_samples" in summary.get("results", {}):
        r = summary["results"]
        print(f"\nResults:")
        print(f"  Total samples: {r['total_samples']}")
        if r.get("source_breakdown"):
            for src, cnt in r["source_breakdown"].items():
                print(f"    {src}: {cnt}")
        print(f"  Sample success rate: {r['sample_success_rate']*100:.1f}%")
        if r.get("prompt_success_rate") != "N/A":
            print(f"  Prompts with >= 1 success: "
                  f"{r['prompts_with_at_least_one_yes']}/{r['n_prompts']}")
            print(f"  Prompt success rate: {r['prompt_success_rate']*100:.1f}%")


if __name__ == "__main__":
    main()
