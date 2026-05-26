#!/usr/bin/env python3
"""Main experiment runner for vLLM tree exploration and greedy baseline evaluation.

Usage:
    python run_experiment.py --model-name llama3_1b --behaviors-csv /path/to/behaviors.csv

Results are saved to: results/{model_name}/
"""

import os
import sys
import argparse
import json
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Optional, Tuple

import yaml


DEFAULT_CUDA_VISIBLE_DEVICES = "2,3,4,5"
DEFAULT_CATEGORIES = ["contextual", "standard"]
CONFIG_SECTION_KEYS = {"classifier", "generation", "categories"}
STEP_SEPARATOR = "=" * 60


def parse_args():
    parser = argparse.ArgumentParser(
        description='Run vLLM tree exploration and greedy baseline experiments'
    )

    # Required arguments
    parser.add_argument(
        '--model-name', '-m',
        required=True,
        help='Model name from config/models.yaml (e.g., llama3_1b, llama3_8b)'
    )
    parser.add_argument(
        '--behaviors-csv', '-b',
        required=True,
        help='Path to HarmBench behaviors CSV file'
    )

    # Output arguments
    parser.add_argument(
        '--output-dir', '-o',
        default='./results',
        help='Base output directory (default: ./results)'
    )
    parser.add_argument(
        '--output-name',
        type=str,
        default=None,
        help='Optional output subdirectory name (default: model name)'
    )
    parser.add_argument(
        '--config-file', '-c',
        default='./config/models.yaml',
        help='Path to models config file'
    )

    # Behavior selection
    parser.add_argument(
        '--categories',
        nargs='+',
        default=None,
        help='Behavior categories to process (default: from config)'
    )
    parser.add_argument(
        '--start-idx',
        type=int,
        default=0,
        help='Start index for behaviors'
    )
    parser.add_argument(
        '--max-count',
        type=int,
        default=None,
        help='Maximum number of behaviors to process'
    )

    # Pipeline control
    parser.add_argument(
        '--skip-tree',
        action='store_true',
        help='Skip tree exploration step'
    )
    parser.add_argument(
        '--tree-dir',
        type=str,
        help='Path to existing tree directory (use with --skip-tree)'
    )
    parser.add_argument(
        '--skip-continuation',
        action='store_true',
        help='Skip continuation generation step'
    )
    parser.add_argument(
        '--skip-greedy',
        action='store_true',
        help='Skip greedy baseline generation step'
    )
    parser.add_argument(
        '--skip-classifier',
        action='store_true',
        help='Skip classifier step'
    )

    # Tree exploration parameters
    parser.add_argument(
        '--exploration-depths',
        nargs='+',
        type=int,
        default=None,
        help='Exploration depths (default: from config)'
    )
    parser.add_argument(
        '--k',
        type=int,
        default=None,
        help='Top-k tokens per step (default: from config)'
    )
    parser.add_argument(
        '--temperature',
        type=float,
        default=None,
        help='Sampling temperature (default: from config)'
    )

    # Continuation parameters
    parser.add_argument(
        '--max-tokens',
        type=int,
        default=None,
        help='Max tokens for continuation (default: from config)'
    )
    parser.add_argument(
        '--batch-size',
        type=int,
        default=None,
        help='Batch size for processing (default: from config)'
    )
    parser.add_argument(
        '--top-p',
        type=float,
        default=None,
        help='Top-p sampling for continuation (default: from config)'
    )
    parser.add_argument(
        '--seed',
        type=int,
        default=42,
        help='Random seed for reproducible continuation sampling'
    )
    parser.add_argument(
        '--num-samples',
        type=int,
        default=1,
        help='Number of samples per leaf for continuation (default: 1)'
    )

    parser.add_argument(
        '--tree-max-depth',
        type=int,
        default=None,
        help='Maximum tree depth for vLLM tree exploration (default: from config or 512)'
    )
    parser.add_argument(
        '--max-logprobs',
        type=int,
        default=None,
        help='Maximum vLLM logprobs to request during tree exploration'
    )

    # Resume
    parser.add_argument(
        '--resume',
        action='store_true',
        help='Resume from checkpoint if exists'
    )

    # GPU
    parser.add_argument(
        '--gpu-ids',
        type=str,
        default=None,
        help='GPU IDs to use (e.g., "0,1")'
    )

    # System prompt
    parser.add_argument(
        '--system-prompt-suffix',
        type=str,
        default=None,
        help='Path to file with additional system prompt text to append'
    )

    return parser.parse_args()


def load_config(config_path: str) -> Dict:
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def set_cuda_visible_devices(gpu_ids: Optional[str]) -> None:
    """Apply explicit GPU ids or the historical evaluation default."""
    if gpu_ids:
        os.environ["CUDA_VISIBLE_DEVICES"] = gpu_ids
    elif "CUDA_VISIBLE_DEVICES" not in os.environ:
        os.environ["CUDA_VISIBLE_DEVICES"] = DEFAULT_CUDA_VISIBLE_DEVICES


def available_model_names(config: Dict) -> List[str]:
    """Return model entries while excluding config-only sections."""
    return [key for key in config if key not in CONFIG_SECTION_KEYS]


def config_value(args, gen_config: Dict, name: str, default):
    """Prefer an argparse override, then the generation config, then a fallback."""
    value = getattr(args, name, None)
    if value is not None:
        return value
    return gen_config.get(name, default)


def setup_output_dir(base_dir: str, model_name: str, output_name: Optional[str] = None) -> Path:
    """Create output directory for model (no timestamp)."""
    dir_name = output_name or model_name
    output_dir = Path(base_dir) / dir_name
    output_dir.mkdir(parents=True, exist_ok=True)

    # Create subdirectories
    (output_dir / "trees").mkdir(exist_ok=True)

    return output_dir


def run_command(cmd: list, description: str):
    """Run a command as subprocess."""
    import subprocess

    print(f"\n{STEP_SEPARATOR}")
    print(f"[STEP] {description}")
    print(STEP_SEPARATOR)
    print(f"Command: {' '.join(cmd)}\n", flush=True)

    env = os.environ.copy()
    env['PYTHONUNBUFFERED'] = '1'
    result = subprocess.run(cmd, env=env)
    if result.returncode != 0:
        raise SystemExit(result.returncode)


def run_tree_pipeline(
    model_config: Dict,
    gen_config: Dict,
    output_dir: Path,
    args,
    categories: List[str]
) -> Tuple[str, str]:
    """Run tree exploration and continuation pipeline."""
    tree_dir = Path(args.tree_dir) if args.tree_dir else output_dir / "trees"
    continued_path = str(output_dir / "continued.csv")
    script_dir = Path(__file__).parent
    k = config_value(args, gen_config, 'k', 5)

    # Tree exploration (subprocess)
    if not args.skip_tree:
        exploration_depths = args.exploration_depths or gen_config.get(
            'exploration_depths', [3, 3, 3, 3, 3, 3, 2]
        )
        temperature = args.temperature or gen_config.get('temperature', 0.7)
        top_p = args.top_p or gen_config.get('top_p', 0.9)
        min_p = gen_config.get('min_p', 0.0)
        tree_batch_size = gen_config.get('tree_batch_size', 128)
        tree_max_logprobs = max(
            k if k > 0 else 0,
            args.max_logprobs if args.max_logprobs is not None else gen_config.get('tree_max_logprobs', 200),
            200,
        )
        tree_max_depth = (
            args.tree_max_depth
            if args.tree_max_depth is not None
            else gen_config.get('tree_max_depth', 512)
        )

        tree_cmd = [
            sys.executable, str(script_dir / 'modules' / 'tree_explorer_vllm.py'),
            '--behaviors-csv', args.behaviors_csv,
            '--model-path', model_config['model_path'],
            '--model-type', model_config['model_type'],
            '--dtype', str(model_config.get('dtype', 'float16')),
            '--output-dir', str(tree_dir),
            '--exploration-depths', *[str(d) for d in exploration_depths],
            '--k', str(k),
            '--temperature', str(temperature),
            '--top-p', str(top_p),
            '--min-p', str(min_p),
            '--batch-size', str(tree_batch_size),
            '--start-idx', str(args.start_idx),
            '--num-gpus', str(model_config.get('num_gpus', 2)),
            '--max-model-len', str(model_config.get('max_model_len', 2048)),
            '--gpu-memory-utilization', str(model_config.get('gpu_memory_utilization', 0.7)),
            '--max-depth', str(tree_max_depth),
            '--max-logprobs', str(tree_max_logprobs),
        ]
        if args.max_count is not None:
            tree_cmd.extend(['--max-count', str(args.max_count)])
        tree_cmd.extend(['--categories', *categories])
        if args.system_prompt_suffix:
            tree_cmd.extend(['--system-prompt-suffix', args.system_prompt_suffix])

        # Count expected behaviors from CSV
        import csv
        import glob
        with open(args.behaviors_csv, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            all_behaviors = [r for r in reader if r.get('FunctionalCategory', 'standard') in categories]
        if args.max_count is not None:
            expected = min(args.max_count, len(all_behaviors) - args.start_idx)
        else:
            expected = len(all_behaviors) - args.start_idx
        expected = max(expected, 0)

        run_command(tree_cmd, "Tree Exploration (vLLM)")

        # Check tree file count
        generated = len(glob.glob(str(tree_dir / "*.json")))
        if generated != expected:
            print(f"\n[WARNING] Tree generation: {generated}/{expected} files")
    else:
        print("\n[SKIP] Tree exploration")

    # Continuation + Greedy baseline (subprocess, same model)
    greedy_path = str(output_dir / "greedy.csv")

    if not args.skip_continuation:
        max_tokens = args.max_tokens or gen_config.get('max_tokens', 512)
        batch_size = args.batch_size or gen_config.get('batch_size', 2000)
        cont_temperature = args.temperature or gen_config.get('temperature', 0.6)
        cont_top_p = args.top_p or gen_config.get('top_p', 0.9)
        cont_seed = args.seed  # None if not specified

        cont_cmd = [
            sys.executable, str(script_dir / 'modules' / 'continuation.py'),
            '--input', str(tree_dir),
            '--output', continued_path,
            '--model-path', model_config['model_path'],
            '--model-type', model_config['model_type'],
            '--dtype', str(model_config.get('dtype', 'float16')),
            '--num-gpus', str(model_config.get('num_gpus', 2)),
            '--max-model-len', str(model_config.get('max_model_len', 2048)),
            '--gpu-memory-utilization', str(model_config.get('gpu_memory_utilization', 0.7)),
            '--batch-size', str(batch_size),
            '--max-tokens', str(max_tokens),
            '--temperature', str(cont_temperature),
            '--top-p', str(cont_top_p),
            '--top-k', str(k),
        ]
        if cont_seed is not None:
            cont_cmd.extend(['--seed', str(cont_seed)])
        if args.num_samples > 1:
            cont_cmd.extend(['--num-samples', str(args.num_samples)])

        # Add greedy baseline args (same subprocess, same model)
        if not args.skip_greedy:
            cont_cmd.extend([
                '--behaviors-csv', args.behaviors_csv,
                '--greedy-output', greedy_path,
                '--start-idx', str(args.start_idx),
            ])
            if args.max_count is not None:
                cont_cmd.extend(['--max-count', str(args.max_count)])
            cont_cmd.extend(['--categories', *categories])

        if args.resume:
            cont_cmd.append('--resume')
        if args.system_prompt_suffix:
            cont_cmd.extend(['--system-prompt-suffix', args.system_prompt_suffix])

        run_command(cont_cmd, "Continuation + Greedy Generation")
    else:
        print("\n[SKIP] Continuation generation")

    return continued_path, greedy_path


def run_classifier(
    input_paths: List[str],
    output_paths: List[str],
    classifier_config: Dict,
    tree_dir: str,
    args,
    categories: List[str],
    greedy_alias_path: str = None,
):
    """Run classifier on multiple inputs (subprocess, single model load)."""
    script_dir = Path(__file__).parent
    from modules.classifier_preprocess import preprocess_experiment_classifier_inputs

    continued_input_path = next(
        (path for path in input_paths if Path(path).name == 'continued.csv'),
        None,
    )
    greedy_input_path = next(
        (path for path in input_paths if Path(path).name == 'greedy.csv'),
        None,
    )
    # Baseline files use the same format as greedy (behavior_id column).
    if greedy_alias_path and not greedy_input_path:
        greedy_input_path = greedy_alias_path

    processed_continued_path, processed_greedy_path = preprocess_experiment_classifier_inputs(
        continued_input_path=continued_input_path,
        continued_output_path=(
            str(Path(continued_input_path).with_suffix('.classifier_safe.csv'))
            if continued_input_path else None
        ),
        greedy_input_path=greedy_input_path,
        greedy_output_path=(
            str(Path(greedy_input_path).with_suffix('.classifier_safe.csv'))
            if greedy_input_path else None
        ),
        behaviors_csv=args.behaviors_csv,
        classifier_model_path=classifier_config['model_path'],
        categories=categories,
        start_idx=args.start_idx,
        max_count=args.max_count,
        max_prompt_tokens=2000,
    )

    processed_input_paths = []
    for path in input_paths:
        if continued_input_path and path == continued_input_path:
            processed_input_paths.append(processed_continued_path)
        elif greedy_input_path and path == greedy_input_path:
            processed_input_paths.append(processed_greedy_path)
        else:
            processed_input_paths.append(path)

    cls_cmd = [
        sys.executable, str(script_dir / 'modules' / 'classifier.py'),
        '--input', *processed_input_paths,
        '--output', *output_paths,
        '--model-path', classifier_config['model_path'],
        '--num-gpus', str(classifier_config.get('num_gpus', 2)),
        '--max-model-len', str(classifier_config.get('max_model_len', 2048)),
        '--gpu-memory-utilization', str(classifier_config.get('gpu_memory_utilization', 0.7)),
        '--tree-dir', tree_dir,
        '--behaviors-csv', args.behaviors_csv,
        '--generation-col', 'full_continuation',
        '--batch-size', str(args.batch_size or 2000),
    ]
    if args.resume:
        cls_cmd.append('--resume')

    run_command(cls_cmd, "Classification")


def generate_summary(output_dir: Path, args, config: Dict, categories: List[str]):
    """Generate experiment summary."""
    import pandas as pd

    summary = {
        'experiment': {
            'model_name': args.model_name,
            'timestamp': datetime.now().isoformat(),
            'categories': categories,
        },
        'config': {
            'model': config.get(args.model_name, {}),
            'generation': config.get('generation', {})
        },
        'results': {}
    }

    # Tree results
    classified_path = output_dir / "classified.csv"
    if classified_path.exists():
        df = pd.read_csv(classified_path)
        yes_count = sum(1 for r in df['classifier_result'] if str(r).lower() == 'yes')
        total = len(df)
        summary['results']['tree'] = {
            'total_paths': total,
            'yes_count': yes_count,
            'no_count': total - yes_count,
            'success_rate': yes_count / total if total > 0 else 0
        }

    # Greedy results
    greedy_classified_path = output_dir / "greedy_classified.csv"
    if greedy_classified_path.exists():
        df = pd.read_csv(greedy_classified_path)
        yes_count = sum(1 for r in df['classifier_result'] if str(r).lower() == 'yes')
        total = len(df)
        summary['results']['greedy'] = {
            'total': total,
            'yes_count': yes_count,
            'no_count': total - yes_count,
            'success_rate': yes_count / total if total > 0 else 0
        }

    # Save summary
    summary_path = output_dir / "summary.json"
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)

    print(f"\nSummary saved to {summary_path}")

    return summary


def main():
    args = parse_args()

    set_cuda_visible_devices(args.gpu_ids)

    # Load config
    print(f"Loading config from {args.config_file}...")
    config = load_config(args.config_file)

    if args.model_name not in config:
        print(f"Error: Model '{args.model_name}' not found in config")
        print(f"Available models: {available_model_names(config)}")
        sys.exit(1)

    model_config = config[args.model_name]
    classifier_config = config.get('classifier', {})
    gen_config = config.get('generation', {})

    print(f"Model: {args.model_name}")
    print(f"Model path: {model_config['model_path']}")
    print(f"Model type: {model_config['model_type']}")

    # Setup output directory
    output_dir = setup_output_dir(args.output_dir, args.model_name, args.output_name)
    print(f"Output directory: {output_dir}")

    # Categories for tree exploration
    categories = args.categories or config.get('categories', DEFAULT_CATEGORIES)
    print(f"Categories: {categories}")

    continued_path = str(output_dir / "continued.csv")
    greedy_path = str(output_dir / "greedy.csv")
    if not args.skip_tree or not args.skip_continuation:
        result = run_tree_pipeline(
            model_config=model_config,
            gen_config=gen_config,
            output_dir=output_dir,
            args=args,
            categories=categories
        )
        continued_path, greedy_path = result

    if not args.skip_classifier:
        input_paths = []
        output_paths = []

        if continued_path and os.path.exists(continued_path):
            input_paths.append(continued_path)
            output_paths.append(str(output_dir / "classified.csv"))

        if not args.skip_greedy and greedy_path and os.path.exists(greedy_path):
            input_paths.append(greedy_path)
            output_paths.append(str(output_dir / "greedy_classified.csv"))

        if input_paths:
            tree_dir = str(Path(args.tree_dir) if args.tree_dir else output_dir / "trees")
            run_classifier(
                input_paths=input_paths,
                output_paths=output_paths,
                classifier_config=classifier_config,
                tree_dir=tree_dir,
                args=args,
                categories=categories,
            )

    # Generate summary
    summary = generate_summary(output_dir, args, config, categories)

    # Final output
    print("\n" + STEP_SEPARATOR)
    print("EXPERIMENT COMPLETE")
    print(STEP_SEPARATOR)
    print(f"Results saved to: {output_dir}")

    if 'tree' in summary.get('results', {}):
        tree_results = summary['results']['tree']
        print(f"\nTree Results:")
        print(f"  Total paths: {tree_results['total_paths']}")
        print(f"  Success rate: {tree_results['success_rate']*100:.1f}%")

    if 'greedy' in summary.get('results', {}):
        greedy_results = summary['results']['greedy']
        print(f"\nGreedy Baseline Results:")
        print(f"  Total: {greedy_results['total']}")
        print(f"  Success rate: {greedy_results['success_rate']*100:.1f}%")


if __name__ == "__main__":
    main()
