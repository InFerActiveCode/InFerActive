#!/usr/bin/env python3
"""Full pipeline for hybrid random-walk tree continuation experiments.

Pipeline:
  1. Resume existing DFS trees into hybrid walk-count trees.
  2. Generate continuations from hybrid tree leaf walk_counts.
  3. Run the HarmBench classifier.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml


DEFAULT_CUDA_VISIBLE_DEVICES = "2,3,4,5"
DEFAULT_CATEGORIES = ["contextual", "standard"]
CONFIG_SECTION_KEYS = {"classifier", "generation", "categories"}
STEP_SEPARATOR = "=" * 60


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run hybrid tree random-walk continuation + classifier"
    )

    parser.add_argument("--model-name", "-m", required=True, help="Model key in config")
    parser.add_argument("--behaviors-csv", "-b", required=True, help="HarmBench behaviors CSV")
    parser.add_argument(
        "--input-tree-dir",
        required=True,
        help="Existing DFS tree JSON directory used as the hybrid source",
    )

    parser.add_argument("--output-dir", "-o", default="./results")
    parser.add_argument("--output-name", default=None)
    parser.add_argument("--config-file", "-c", default="./config/models.yaml")
    parser.add_argument("--categories", nargs="+", default=None)
    parser.add_argument("--start-idx", type=int, default=0)
    parser.add_argument("--max-count", type=int, default=None)

    parser.add_argument("--target-depth", type=int, default=20)
    parser.add_argument("--num-walks", "-n", type=int, default=1000)
    parser.add_argument("--num-samples", type=int, default=1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--walk-batch-size", type=int, default=1000)
    parser.add_argument(
        "--preexpand-min-depth",
        type=int,
        default=None,
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--preexpand-max-nodes",
        type=int,
        default=10000,
        help="Unused legacy option; preexpand is disabled",
    )
    parser.add_argument(
        "--max-batch-size",
        type=int,
        default=None,
        help="Maximum vLLM batch size for tree expansion and continuation/classifier defaults",
    )

    parser.add_argument("--k", type=int, default=None)
    parser.add_argument("--temperature", type=float, default=None)
    parser.add_argument("--top-p", type=float, default=None)
    parser.add_argument("--min-p", type=float, default=None)
    parser.add_argument("--max-logprobs", type=int, default=None)

    parser.add_argument("--max-tokens", type=int, default=None)
    parser.add_argument("--continuation-batch-size", type=int, default=None)
    parser.add_argument("--top-k", type=int, default=None)
    parser.add_argument("--classifier-batch-size", type=int, default=None)

    parser.add_argument("--skip-hybrid-tree", action="store_true")
    parser.add_argument("--hybrid-tree-dir", default=None)
    parser.add_argument("--skip-continuation", action="store_true")
    parser.add_argument("--continued-csv", default=None)
    parser.add_argument("--skip-classifier", action="store_true")
    parser.add_argument("--resume", action="store_true")

    parser.add_argument("--gpu-ids", default=None)
    parser.add_argument("--system-prompt-suffix", default=None)

    return parser.parse_args()


def load_config(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def set_cuda_visible_devices(gpu_ids: Optional[str]) -> None:
    if gpu_ids:
        os.environ["CUDA_VISIBLE_DEVICES"] = gpu_ids
    elif "CUDA_VISIBLE_DEVICES" not in os.environ:
        os.environ["CUDA_VISIBLE_DEVICES"] = DEFAULT_CUDA_VISIBLE_DEVICES


def available_model_names(config: Dict[str, Any]) -> List[str]:
    return [key for key in config if key not in CONFIG_SECTION_KEYS]


def validate_args(args: argparse.Namespace) -> None:
    positive_int_fields = [
        "target_depth",
        "num_walks",
        "num_samples",
        "walk_batch_size",
    ]
    optional_positive_int_fields = [
        "max_batch_size",
        "max_logprobs",
        "max_tokens",
        "continuation_batch_size",
        "classifier_batch_size",
    ]

    for field in positive_int_fields:
        if int(getattr(args, field)) < 1:
            raise ValueError(f"{field} must be >= 1")

    for field in optional_positive_int_fields:
        value = getattr(args, field)
        if value is not None and int(value) < 1:
            raise ValueError(f"{field} must be >= 1")

    if int(args.preexpand_max_nodes) < 0:
        raise ValueError("preexpand_max_nodes must be >= 0")


def run_command(cmd: List[str], description: str) -> int:
    print(f"\n{STEP_SEPARATOR}")
    print(f"[STEP] {description}")
    print(STEP_SEPARATOR)
    print(f"Command: {' '.join(cmd)}\n", flush=True)

    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"
    result = subprocess.run(cmd, env=env)
    return int(result.returncode)


def generation_value(args: argparse.Namespace, gen_config: Dict[str, Any], name: str, default: Any) -> Any:
    value = getattr(args, name, None)
    if value is not None:
        return value
    return gen_config.get(name, default)


def run_hybrid_tree_step(
    model_config: Dict[str, Any],
    gen_config: Dict[str, Any],
    input_tree_dir: Path,
    hybrid_tree_dir: Path,
    categories: List[str],
    args: argparse.Namespace,
) -> bool:
    script_dir = Path(__file__).parent

    k = generation_value(args, gen_config, "k", 0)
    temperature = generation_value(args, gen_config, "temperature", 0.6)
    top_p = generation_value(args, gen_config, "top_p", 0.9)
    min_p = generation_value(args, gen_config, "min_p", 0.0)
    max_logprobs = args.max_logprobs if args.max_logprobs is not None else gen_config.get("max_logprobs", 200)
    max_batch_size = (
        args.max_batch_size
        if args.max_batch_size is not None
        else gen_config.get("hybrid_tree_batch_size", gen_config.get("tree_batch_size", 128))
    )

    cmd = [
        sys.executable,
        str(script_dir / "modules" / "hybrid_random_tree_explorer.py"),
        "--input-tree-dir",
        str(input_tree_dir),
        "--output-dir",
        str(hybrid_tree_dir),
        "--behaviors-csv",
        args.behaviors_csv,
        "--model-path",
        model_config["model_path"],
        "--model-type",
        model_config["model_type"],
        "--dtype",
        str(model_config.get("dtype", "float16")),
        "--num-gpus",
        str(model_config.get("num_gpus", 2)),
        "--max-model-len",
        str(model_config.get("max_model_len", 2048)),
        "--gpu-memory-utilization",
        str(model_config.get("gpu_memory_utilization", 0.7)),
        "--k",
        str(k),
        "--temperature",
        str(temperature),
        "--top-p",
        str(top_p),
        "--min-p",
        str(min_p),
        "--max-logprobs",
        str(max_logprobs),
        "--target-depth",
        str(args.target_depth),
        "--num-walks",
        str(args.num_walks),
        "--num-samples",
        str(args.num_samples),
        "--seed",
        str(args.seed),
        "--walk-batch-size",
        str(args.walk_batch_size),
        "--preexpand-max-nodes",
        str(args.preexpand_max_nodes),
        "--max-batch-size",
        str(max_batch_size),
        "--start-idx",
        str(args.start_idx),
    ]
    if args.max_count is not None:
        cmd.extend(["--max-count", str(args.max_count)])
    if categories:
        cmd.extend(["--categories", *categories])
    if args.resume:
        cmd.append("--resume")
    if args.system_prompt_suffix:
        cmd.extend(["--system-prompt-suffix", args.system_prompt_suffix])

    return run_command(cmd, "Hybrid Tree Walk Generation") == 0


def run_continuation_step(
    model_config: Dict[str, Any],
    gen_config: Dict[str, Any],
    hybrid_tree_dir: Path,
    output_csv: Path,
    args: argparse.Namespace,
) -> bool:
    script_dir = Path(__file__).parent

    max_tokens = args.max_tokens if args.max_tokens is not None else gen_config.get("max_tokens", 200)
    temperature = generation_value(args, gen_config, "temperature", 0.6)
    top_p = generation_value(args, gen_config, "top_p", 0.9)
    top_k = args.top_k if args.top_k is not None else gen_config.get("top_k", gen_config.get("k", -1))
    batch_size = (
        args.continuation_batch_size
        if args.continuation_batch_size is not None
        else (
            args.max_batch_size
            if args.max_batch_size is not None
            else gen_config.get("batch_size", 2000)
        )
    )

    cmd = [
        sys.executable,
        str(script_dir / "modules" / "hybrid_walk_continuation.py"),
        "--tree-dir",
        str(hybrid_tree_dir),
        "--output",
        str(output_csv),
        "--model-path",
        model_config["model_path"],
        "--model-type",
        model_config["model_type"],
        "--dtype",
        str(model_config.get("dtype", "float16")),
        "--num-gpus",
        str(model_config.get("num_gpus", 2)),
        "--max-model-len",
        str(model_config.get("max_model_len", 2048)),
        "--gpu-memory-utilization",
        str(model_config.get("gpu_memory_utilization", 0.7)),
        "--max-tokens",
        str(max_tokens),
        "--temperature",
        str(temperature),
        "--top-p",
        str(top_p),
        "--top-k",
        str(top_k),
        "--batch-size",
        str(batch_size),
        "--num-samples",
        str(args.num_samples),
    ]
    if args.resume:
        cmd.append("--resume")
    if args.system_prompt_suffix:
        cmd.extend(["--system-prompt-suffix", args.system_prompt_suffix])

    return run_command(cmd, "Hybrid Walk Continuation") == 0


def run_classifier_step(
    input_csv: Path,
    output_csv: Path,
    classifier_config: Dict[str, Any],
    args: argparse.Namespace,
    categories: List[str],
) -> bool:
    script_dir = Path(__file__).parent

    safe_candidate = str(input_csv.with_suffix(".classifier_safe.csv"))
    if Path(safe_candidate).exists() and args.resume:
        print(f"[classifier_preprocess] Reusing existing safe file: {safe_candidate}")
        safe_input_path = safe_candidate
    else:
        try:
            from modules.classifier_preprocess import preprocess_experiment_classifier_inputs

            safe_input_path, _ = preprocess_experiment_classifier_inputs(
                continued_input_path=str(input_csv),
                continued_output_path=safe_candidate,
                greedy_input_path=None,
                greedy_output_path=None,
                behaviors_csv=args.behaviors_csv,
                classifier_model_path=classifier_config["model_path"],
                categories=categories,
                start_idx=args.start_idx,
                max_count=args.max_count,
            )
        except Exception as exc:
            print(f"[WARN] Preprocessing failed ({exc}), using raw input")
            safe_input_path = str(input_csv)

    batch_size = (
        args.classifier_batch_size
        if args.classifier_batch_size is not None
        else (
            args.max_batch_size
            if args.max_batch_size is not None
            else 2000
        )
    )

    cmd = [
        sys.executable,
        str(script_dir / "modules" / "classifier.py"),
        "--input",
        str(safe_input_path),
        "--output",
        str(output_csv),
        "--model-path",
        classifier_config["model_path"],
        "--num-gpus",
        str(classifier_config.get("num_gpus", 2)),
        "--max-model-len",
        str(classifier_config.get("max_model_len", 2048)),
        "--gpu-memory-utilization",
        str(classifier_config.get("gpu_memory_utilization", 0.7)),
        "--behaviors-csv",
        args.behaviors_csv,
        "--generation-col",
        "full_continuation",
        "--batch-size",
        str(batch_size),
    ]
    if args.resume:
        cmd.append("--resume")

    return run_command(cmd, "Classification") == 0


def generate_summary(
    output_dir: Path,
    continuation_path: Path,
    classified_path: Path,
    args: argparse.Namespace,
    config: Dict[str, Any],
    categories: List[str],
) -> Dict[str, Any]:
    summary: Dict[str, Any] = {
        "experiment": {
            "type": "hybrid_random_continuation",
            "model_name": args.model_name,
            "timestamp": datetime.now().isoformat(),
            "categories": categories,
            "target_depth": args.target_depth,
            "num_walks": args.num_walks,
            "num_samples": args.num_samples,
            "seed": args.seed,
        },
        "config": {
            "model": config.get(args.model_name, {}),
            "generation": config.get("generation", {}),
        },
        "results": {},
    }

    try:
        import pandas as pd

        if continuation_path.exists():
            df = pd.read_csv(continuation_path, low_memory=False)
            if {"file_name", "leaf_id", "run_seed", "walk_count"}.issubset(df.columns):
                walk_count_sum = int(
                    df.drop_duplicates(["file_name", "leaf_id", "run_seed"])["walk_count"].sum()
                )
            else:
                walk_count_sum = 0
            summary["results"]["continuation"] = {
                "total_rows": int(len(df)),
                "n_prompts": int(df["file_name"].nunique()) if "file_name" in df else 0,
                "walk_count_sum": walk_count_sum,
            }

        if classified_path.exists():
            df = pd.read_csv(classified_path, low_memory=False)
            yes_count = sum(1 for result in df["classifier_result"] if str(result).lower() == "yes")
            total = len(df)
            prompt_success = df.groupby("file_name")["classifier_result"].apply(
                lambda values: any(str(value).lower() == "yes" for value in values)
            )
            summary["results"]["classifier"] = {
                "total_samples": int(total),
                "yes_count": int(yes_count),
                "no_count": int(total - yes_count),
                "sample_success_rate": yes_count / total if total else 0.0,
                "prompts_with_at_least_one_yes": int(prompt_success.sum()),
                "n_prompts": int(df["file_name"].nunique()),
                "prompt_success_rate": (
                    float(prompt_success.mean()) if len(prompt_success) else 0.0
                ),
            }
    except Exception as exc:
        summary["summary_error"] = str(exc)

    summary_path = output_dir / "summary.json"
    with open(summary_path, "w", encoding="utf-8") as handle:
        json.dump(summary, handle, ensure_ascii=False, indent=2)
    print(f"\nSummary saved to {summary_path}")
    return summary


def main() -> None:
    args = parse_args()
    validate_args(args)

    set_cuda_visible_devices(args.gpu_ids)

    config = load_config(args.config_file)
    if args.model_name not in config:
        print(f"Error: model '{args.model_name}' not found. Available: {available_model_names(config)}")
        sys.exit(1)

    model_config = config[args.model_name]
    classifier_config = config.get("classifier", {})
    gen_config = config.get("generation", {})
    categories = args.categories or config.get("categories", DEFAULT_CATEGORIES)

    dir_name = args.output_name or args.model_name
    output_dir = Path(args.output_dir) / dir_name / "hybrid_random_continuation"
    output_dir.mkdir(parents=True, exist_ok=True)

    input_tree_dir = Path(args.input_tree_dir)
    hybrid_tree_dir = Path(args.hybrid_tree_dir) if args.hybrid_tree_dir else output_dir / "hybrid_trees"
    hybrid_tree_dir.mkdir(parents=True, exist_ok=True)

    continuation_csv = (
        Path(args.continued_csv)
        if args.continued_csv
        else output_dir / "continued_with_hybrid_walk.csv"
    )
    classified_csv = output_dir / "classified.csv"

    print(f"Model: {args.model_name}")
    print(f"Input trees: {input_tree_dir}")
    print(f"Hybrid trees: {hybrid_tree_dir}")
    print(f"Output: {output_dir}")
    print(f"Walks: {args.num_walks} x {args.num_samples} seed samples")

    if not args.skip_hybrid_tree:
        if not run_hybrid_tree_step(
            model_config=model_config,
            gen_config=gen_config,
            input_tree_dir=input_tree_dir,
            hybrid_tree_dir=hybrid_tree_dir,
            categories=categories,
            args=args,
        ):
            print("\n[ABORT] Hybrid tree generation failed")
            sys.exit(1)
    else:
        print("\n[SKIP] Hybrid tree generation")

    if not args.skip_continuation:
        if not run_continuation_step(
            model_config=model_config,
            gen_config=gen_config,
            hybrid_tree_dir=hybrid_tree_dir,
            output_csv=continuation_csv,
            args=args,
        ):
            print("\n[ABORT] Hybrid continuation failed")
            sys.exit(1)
    else:
        print("\n[SKIP] Hybrid continuation")

    if not args.skip_classifier and continuation_csv.exists():
        if not run_classifier_step(
            input_csv=continuation_csv,
            output_csv=classified_csv,
            classifier_config=classifier_config,
            args=args,
            categories=categories,
        ):
            print("\n[ABORT] Classification failed")
            sys.exit(1)
    else:
        print("\n[SKIP] Classification")

    summary = generate_summary(
        output_dir=output_dir,
        continuation_path=continuation_csv,
        classified_path=classified_csv,
        args=args,
        config=config,
        categories=categories,
    )

    print("\n" + STEP_SEPARATOR)
    print("HYBRID RANDOM CONTINUATION PIPELINE COMPLETE")
    print(STEP_SEPARATOR)
    print(f"Results saved to: {output_dir}")

    classifier_summary = summary.get("results", {}).get("classifier")
    if classifier_summary:
        print(f"  Samples: {classifier_summary['total_samples']}")
        print(f"  Sample success rate: {classifier_summary['sample_success_rate'] * 100:.1f}%")
        print(
            "  Prompts with >= 1 success: "
            f"{classifier_summary['prompts_with_at_least_one_yes']}/"
            f"{classifier_summary['n_prompts']}"
        )


if __name__ == "__main__":
    main()
