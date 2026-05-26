#!/usr/bin/env python3
"""
Random sampling experiment runner - generates n samples per prompt without tree exploration.

Usage:
    python run_random_sampling.py --model-name llama3_1b --behaviors-csv /path/to/behaviors.csv --n-samples 10

Results are saved to the selected output directory.
"""

import os
import sys
import argparse
import hashlib
import json
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Optional, Tuple

import yaml


DEFAULT_CATEGORIES = ["contextual", "standard"]
CONFIG_SECTION_KEYS = {"classifier", "generation", "categories"}
STEP_SEPARATOR = "=" * 60


def parse_args():
    parser = argparse.ArgumentParser(
        description='Run random sampling experiments (n samples per prompt)'
    )

    # Required arguments
    parser.add_argument(
        '--model-name', '-m',
        required=True,
        help='Model name from config/models.yaml'
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
        '--config-file', '-c',
        default='./config/models.yaml',
        help='Path to models config file'
    )

    # Behavior selection
    parser.add_argument(
        '--categories',
        nargs='+',
        default=None,
        help='Behavior categories to process'
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

    # Sampling parameters
    parser.add_argument(
        '--n-samples', '-n',
        type=int,
        default=10,
        help='Number of samples per prompt (default: 10)'
    )
    parser.add_argument(
        '--temperature',
        type=float,
        default=None,
        help='Sampling temperature (default: from config or 1.0)'
    )
    parser.add_argument(
        '--top-p',
        type=float,
        default=None,
        help='Top-p sampling (default: from config or 0.9)'
    )
    parser.add_argument(
        '--k',
        type=int,
        default=None,
        help='Top-k sampling (default: from config)'
    )
    parser.add_argument(
        '--min-p',
        type=float,
        default=None,
        help='Min-p sampling (default: from config or 0.0)'
    )
    parser.add_argument(
        '--max-tokens',
        type=int,
        default=None,
        help='Max tokens for generation (default: from config)'
    )
    parser.add_argument(
        '--seed',
        type=int,
        default=None,
        help='Random seed for reproducibility'
    )
    parser.add_argument(
        '--batch-size',
        type=int,
        default=None,
        help='Batch size (prompts per batch)'
    )

    # Unique top-up sampling
    parser.add_argument(
        '--ensure-unique',
        action='store_true',
        help='Keep sampling only prompts with fewer than n unique continuations'
    )
    parser.add_argument(
        '--unique-retry-samples',
        type=int,
        default=None,
        help='Samples to request per incomplete prompt per unique top-up round (default: --n-samples)'
    )
    parser.add_argument(
        '--unique-max-rounds',
        type=int,
        default=20,
        help='Maximum unique top-up rounds (default: 20)'
    )
    parser.add_argument(
        '--unique-stall-rounds',
        type=int,
        default=3,
        help='Stop if this many consecutive unique rounds accept no new samples (default: 3)'
    )
    parser.add_argument(
        '--unique-base-seed',
        type=int,
        default=None,
        help='Base seed for unique top-up rounds (default: --seed, or 42)'
    )
    parser.add_argument(
        '--keep-round-files',
        action='store_true',
        help='Keep temporary per-round sampling CSVs for debugging'
    )
    parser.add_argument(
        '--unique-skip-behavior-ids-file',
        type=str,
        default=None,
        help='File of BehaviorIDs to exclude from unique top-up sampling and final samples.csv'
    )
    parser.add_argument(
        '--unique-allow-partial',
        action='store_true',
        help='If some prompts do not reach n unique samples, write the partial unique samples.csv and continue to classifier'
    )

    # Pipeline control
    parser.add_argument(
        '--skip-generation',
        action='store_true',
        help='Skip generation step (use existing samples.csv)'
    )
    parser.add_argument(
        '--skip-classifier',
        action='store_true',
        help='Skip classifier step'
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

    return parser.parse_args()


def load_config(config_path: str) -> Dict:
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def set_cuda_visible_devices(gpu_ids: Optional[str]) -> None:
    if gpu_ids:
        os.environ["CUDA_VISIBLE_DEVICES"] = gpu_ids


def available_model_names(config: Dict) -> List[str]:
    return [key for key in config if key not in CONFIG_SECTION_KEYS]


def config_value(args, gen_config: Dict, name: str, default):
    value = getattr(args, name, None)
    if value is not None:
        return value
    return gen_config.get(name, default)


def setup_output_dir(base_dir: str, model_name: str) -> Path:
    """Create output directory."""
    output_dir = Path(base_dir) / model_name / "random"
    output_dir.mkdir(parents=True, exist_ok=True)
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
    subprocess.run(cmd, env=env)


def run_random_sampling(
    model_config: Dict,
    gen_config: Dict,
    output_dir: Path,
    args,
    categories: List[str]
) -> str:
    """Run random sampling generation."""
    output_path = str(output_dir / "samples.csv")
    script_dir = Path(__file__).parent

    n_samples = args.n_samples
    temperature = args.temperature or gen_config.get('temperature', 1.0)
    top_p = args.top_p or gen_config.get('top_p', 0.9)
    top_k = config_value(args, gen_config, 'k', 5)
    min_p = config_value(args, gen_config, 'min_p', 0.0)
    max_tokens = args.max_tokens or gen_config.get('max_tokens', 512)

    cmd = [
        sys.executable, str(script_dir / 'modules' / 'random_sampler.py'),
        '--behaviors-csv', args.behaviors_csv,
        '--output', output_path,
        '--model-path', model_config['model_path'],
        '--model-type', model_config['model_type'],
        '--dtype', str(model_config.get('dtype', 'float16')),
        '--num-gpus', str(model_config.get('num_gpus', 2)),
        '--max-model-len', str(model_config.get('max_model_len', 2048)),
        '--gpu-memory-utilization', str(model_config.get('gpu_memory_utilization', 0.7)),
        '--n-samples', str(n_samples),
        '--temperature', str(temperature),
        '--top-p', str(top_p),
        '--top-k', str(top_k),
        '--min-p', str(min_p),
        '--max-tokens', str(max_tokens),
        '--start-idx', str(args.start_idx),
        '--categories', *categories,
    ]

    if args.max_count:
        cmd.extend(['--max-count', str(args.max_count)])
    if args.seed is not None:
        cmd.extend(['--seed', str(args.seed)])
    if args.resume:
        cmd.append('--resume')

    run_command(cmd, f"Random Sampling (n={n_samples})")

    return output_path


def _normalize_text(value) -> str:
    if value is None:
        return ""
    try:
        import pandas as pd
        if pd.isna(value):
            return ""
    except (TypeError, ValueError):
        pass
    return str(value).strip()


def _hash_text(text: str) -> str:
    return hashlib.sha256(text.encode('utf-8')).hexdigest()


def _ensure_state_prompt(
    state: Dict,
    prompt_idx: int,
    behavior: Dict,
    target_unique: int,
) -> Dict:
    per_prompt = state.setdefault('per_prompt', {})
    key = str(prompt_idx)
    if key not in per_prompt:
        per_prompt[key] = {
            'behavior_id': behavior.get('BehaviorID', f'behavior_{prompt_idx}'),
            'target_unique': target_unique,
            'current_unique': 0,
            'generated_attempts': 0,
            'unique_accepted': 0,
            'duplicates_discarded': 0,
            'over_target_discarded': 0,
            'imported_rows': 0,
            'imported_unique': 0,
            'imported_duplicates_discarded': 0,
            'imported_over_target_discarded': 0,
            'rounds_attempted': 0,
            'done': False,
            'last_seed': None,
        }
    return per_prompt[key]


def _init_unique_state(
    state_path: Path,
    behaviors: List[Dict],
    target_unique: int,
    args,
) -> Dict:
    if state_path.exists():
        with open(state_path, 'r') as f:
            state = json.load(f)
    else:
        state = {
            'target_unique_per_prompt': target_unique,
            'rounds_completed': 0,
            'total_generated_attempts': 0,
            'total_unique_accepted': 0,
            'total_duplicates_discarded': 0,
            'total_over_target_discarded': 0,
            'total_imported_rows': 0,
            'total_imported_unique': 0,
            'total_imported_duplicates_discarded': 0,
            'total_imported_over_target_discarded': 0,
            'rounds': [],
            'config': {
                'model_name': args.model_name,
                'behaviors_csv': args.behaviors_csv,
                'categories': args.categories,
                'start_idx': args.start_idx,
                'max_count': args.max_count,
            },
            'per_prompt': {},
        }

    state['target_unique_per_prompt'] = target_unique
    for prompt_idx, behavior in enumerate(behaviors):
        _ensure_state_prompt(state, prompt_idx, behavior, target_unique)
    return state


def _write_unique_state(state_path: Path, state: Dict):
    tmp_path = state_path.with_suffix(state_path.suffix + '.tmp')
    with open(tmp_path, 'w') as f:
        json.dump(state, f, indent=2)
    tmp_path.replace(state_path)


def _pool_columns() -> List[str]:
    return [
        'prompt_idx',
        'sample_idx',
        'behavior_id',
        'category',
        'full_continuation',
        'count',
        'text_hash',
        'source',
        'source_round',
        'source_seed',
        'source_prompt_idx',
        'source_sample_idx',
    ]


def _round_count_columns(df) -> List[str]:
    return sorted(c for c in df.columns if str(c).startswith('count_round_'))


def _pool_output_columns(pool_df) -> List[str]:
    return _pool_columns() + _round_count_columns(pool_df)


def _normalize_round_count_columns(df):
    import pandas as pd

    for col in _round_count_columns(df):
        df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0).astype(int)
        df.loc[df[col] < 0, col] = 0


def _load_unique_pool(pool_path: Path):
    import pandas as pd

    if not pool_path.exists():
        return pd.DataFrame(columns=_pool_columns())
    pool_df = pd.read_csv(pool_path, low_memory=False)
    for col in _pool_columns():
        if col == 'count' and col not in pool_df.columns:
            pool_df[col] = 1
        elif col not in pool_df.columns:
            pool_df[col] = None
    pool_df['count'] = pd.to_numeric(pool_df['count'], errors='coerce').fillna(1).astype(int)
    pool_df.loc[pool_df['count'] < 1, 'count'] = 1
    _normalize_round_count_columns(pool_df)
    return pool_df[_pool_output_columns(pool_df)].copy()


def _write_unique_pool(pool_path: Path, pool_df):
    pool_df = pool_df.copy()
    for col in _pool_columns():
        if col == 'count' and col not in pool_df.columns:
            pool_df[col] = 1
        elif col not in pool_df.columns:
            pool_df[col] = None
    pool_df['count'] = pool_df['count'].map(_positive_count)
    _normalize_round_count_columns(pool_df)
    pool_path.parent.mkdir(parents=True, exist_ok=True)
    pool_df = pool_df[_pool_output_columns(pool_df)].copy()
    pool_df.to_csv(pool_path, index=False)


def _update_current_unique_counts(state: Dict, pool_df, behaviors: List[Dict], target_unique: int):
    if pool_df.empty:
        counts = {}
    else:
        counts = pool_df.groupby('prompt_idx')['text_hash'].nunique().to_dict()

    for prompt_idx, behavior in enumerate(behaviors):
        entry = _ensure_state_prompt(state, prompt_idx, behavior, target_unique)
        current = int(counts.get(prompt_idx, 0))
        entry['current_unique'] = current
        entry['done'] = current >= target_unique


def _get_failed_prompt_idxs(
    pool_df,
    prompt_count: int,
    target_unique: int,
    skip_prompt_idxs=None,
) -> List[int]:
    if pool_df.empty:
        counts = {}
    else:
        counts = pool_df.groupby('prompt_idx')['text_hash'].nunique().to_dict()
    skip_prompt_idxs = set(skip_prompt_idxs or [])
    return [
        prompt_idx
        for prompt_idx in range(prompt_count)
        if prompt_idx not in skip_prompt_idxs
        and int(counts.get(prompt_idx, 0)) < target_unique
    ]


def _get_max_deficit(
    pool_df,
    prompt_count: int,
    target_unique: int,
    skip_prompt_idxs=None,
) -> int:
    if pool_df.empty:
        counts = {}
    else:
        counts = pool_df.groupby('prompt_idx')['text_hash'].nunique().to_dict()
    skip_prompt_idxs = set(skip_prompt_idxs or [])
    deficits = [
        target_unique - int(counts.get(prompt_idx, 0))
        for prompt_idx in range(prompt_count)
        if prompt_idx not in skip_prompt_idxs
    ]
    return max([d for d in deficits if d > 0], default=0)


def _record_count(entry: Dict, key: str, amount: int):
    entry[key] = int(entry.get(key, 0)) + int(amount)


def _positive_count(value) -> int:
    try:
        count = int(value)
    except (TypeError, ValueError):
        try:
            count = int(float(value))
        except (TypeError, ValueError):
            count = 1
    return max(1, count)


def _load_behavior_ids_file(path: str) -> set:
    if not path:
        return set()

    behavior_ids = set()
    with open(path, 'r') as f:
        for raw_line in f:
            line = raw_line.split('#', 1)[0].strip()
            if not line:
                continue
            if ':' in line:
                line = line.split(':', 1)[0].strip()
            if ',' in line:
                line = line.split(',', 1)[0].strip()
            if line:
                behavior_ids.add(line)
    return behavior_ids


def _import_existing_samples_to_pool(
    samples_path: Path,
    pool_path: Path,
    state: Dict,
    behaviors: List[Dict],
    target_unique: int,
):
    import pandas as pd

    if pool_path.exists() or not samples_path.exists():
        return

    print(f"[UNIQUE] Importing existing samples from {samples_path}")
    df = pd.read_csv(samples_path, low_memory=False)
    if 'prompt_idx' not in df.columns or 'full_continuation' not in df.columns:
        print("[UNIQUE] Existing samples.csv is missing prompt_idx/full_continuation; skipping import")
        return

    df = df.copy()
    df['prompt_idx'] = pd.to_numeric(df['prompt_idx'], errors='coerce')
    df = df[df['prompt_idx'].notna()].copy()
    df['prompt_idx'] = df['prompt_idx'].astype(int)
    df = df[(df['prompt_idx'] >= 0) & (df['prompt_idx'] < len(behaviors))].copy()
    if df.empty:
        print("[UNIQUE] Existing samples.csv has no rows in the selected prompt range")
        return

    if 'sample_idx' not in df.columns:
        df['sample_idx'] = df.groupby('prompt_idx').cumcount()
    if 'behavior_id' not in df.columns:
        df['behavior_id'] = df['prompt_idx'].map(
            {idx: b.get('BehaviorID', f'behavior_{idx}') for idx, b in enumerate(behaviors)}
        )
    if 'category' not in df.columns:
        df['category'] = df['prompt_idx'].map(
            {idx: b.get('FunctionalCategory', 'standard') for idx, b in enumerate(behaviors)}
        )

    df['_normalized_text'] = df['full_continuation'].map(_normalize_text)
    df['text_hash'] = df['_normalized_text'].map(_hash_text)
    if 'count' in df.columns:
        df['count'] = df['count'].map(_positive_count)
    else:
        df['count'] = 1
    round_count_cols = _round_count_columns(df)
    _normalize_round_count_columns(df)
    df['source'] = 'imported_samples_csv'
    df['source_round'] = -1
    df['source_seed'] = None
    df['source_prompt_idx'] = df['prompt_idx']
    df['source_sample_idx'] = df['sample_idx']

    raw_counts = df.groupby('prompt_idx')['count'].sum().to_dict()
    count_by_unique_key = (
        df.groupby(['prompt_idx', 'text_hash'])[['count', *round_count_cols]]
        .sum()
        .reset_index()
        .rename(columns={'count': '_count'})
    )
    unique_df = df.drop_duplicates(['prompt_idx', 'text_hash'], keep='first').copy()
    unique_df = unique_df.drop(columns=['count', *round_count_cols]).merge(
        count_by_unique_key,
        on=['prompt_idx', 'text_hash'],
        how='left',
    )
    unique_df['count'] = unique_df['_count'].map(_positive_count)
    unique_df = unique_df.drop(columns=['_count'])
    if not _round_count_columns(unique_df):
        unique_df['count_round_imported'] = unique_df['count']
    unique_counts = unique_df.groupby('prompt_idx').size().to_dict()
    accepted_df = unique_df.sort_values(['prompt_idx', 'sample_idx']).groupby('prompt_idx').head(target_unique).copy()
    accepted_counts = accepted_df.groupby('prompt_idx').size().to_dict()
    total_imported_attempts = int(df['count'].sum())
    total_imported_duplicate_attempts = int(total_imported_attempts - len(unique_df))

    for prompt_idx, behavior in enumerate(behaviors):
        raw_count = int(raw_counts.get(prompt_idx, 0))
        unique_count = int(unique_counts.get(prompt_idx, 0))
        accepted_count = int(accepted_counts.get(prompt_idx, 0))
        duplicate_count = max(0, raw_count - unique_count)
        over_target_count = max(0, unique_count - accepted_count)

        entry = _ensure_state_prompt(state, prompt_idx, behavior, target_unique)
        _record_count(entry, 'generated_attempts', raw_count)
        _record_count(entry, 'unique_accepted', accepted_count)
        _record_count(entry, 'duplicates_discarded', duplicate_count)
        _record_count(entry, 'over_target_discarded', over_target_count)
        _record_count(entry, 'imported_rows', raw_count)
        _record_count(entry, 'imported_unique', accepted_count)
        _record_count(entry, 'imported_duplicates_discarded', duplicate_count)
        _record_count(entry, 'imported_over_target_discarded', over_target_count)

    state['total_generated_attempts'] = int(state.get('total_generated_attempts', 0)) + total_imported_attempts
    state['total_unique_accepted'] = int(state.get('total_unique_accepted', 0)) + int(len(accepted_df))
    state['total_duplicates_discarded'] = int(state.get('total_duplicates_discarded', 0)) + total_imported_duplicate_attempts
    state['total_over_target_discarded'] = int(state.get('total_over_target_discarded', 0)) + int(len(unique_df) - len(accepted_df))
    state['total_imported_rows'] = int(state.get('total_imported_rows', 0)) + total_imported_attempts
    state['total_imported_unique'] = int(state.get('total_imported_unique', 0)) + int(len(accepted_df))
    state['total_imported_duplicates_discarded'] = (
        int(state.get('total_imported_duplicates_discarded', 0)) + total_imported_duplicate_attempts
    )
    state['total_imported_over_target_discarded'] = (
        int(state.get('total_imported_over_target_discarded', 0)) + int(len(unique_df) - len(accepted_df))
    )

    accepted_df = accepted_df[_pool_output_columns(accepted_df)].copy()
    accepted_df['sample_idx'] = accepted_df.groupby('prompt_idx').cumcount()
    _write_unique_pool(pool_path, accepted_df)
    print(
        f"[UNIQUE] Imported {len(accepted_df)} unique rows "
        f"({total_imported_duplicate_attempts} duplicates, {len(unique_df) - len(accepted_df)} over target discarded)"
    )


def _prepare_round_df(round_df, local_to_original: Dict[int, int], round_idx: int, seed: int):
    import pandas as pd

    round_df = round_df.copy()
    round_df['source_prompt_idx'] = round_df['prompt_idx']
    round_df['source_sample_idx'] = round_df.get('sample_idx', pd.Series(range(len(round_df))))
    round_df['prompt_idx'] = round_df['source_prompt_idx'].map(local_to_original)
    round_df = round_df[round_df['prompt_idx'].notna()].copy()
    round_df['prompt_idx'] = round_df['prompt_idx'].astype(int)
    round_df['source_round'] = round_idx
    round_df['source_seed'] = seed
    round_df['source'] = 'unique_topup'
    round_df['_normalized_text'] = round_df['full_continuation'].map(_normalize_text)
    round_df['text_hash'] = round_df['_normalized_text'].map(_hash_text)
    round_df['count'] = 1
    return round_df


def _merge_round_into_pool(
    pool_df,
    round_df,
    target_unique: int,
    round_idx: int,
) -> Tuple[object, object, Dict[int, Dict[str, int]]]:
    import pandas as pd

    pool_df = pool_df.copy()
    if 'count' not in pool_df.columns:
        pool_df['count'] = 1
    pool_df['count'] = pool_df['count'].map(_positive_count)
    round_count_col = f"count_round_{round_idx:04d}"
    if round_count_col not in pool_df.columns:
        pool_df[round_count_col] = 0
    _normalize_round_count_columns(pool_df)

    accepted_rows = []
    stats: Dict[int, Dict[str, int]] = {}
    pool_key_to_index = {}
    accepted_key_to_pos = {}
    current_counts: Dict[int, int] = {}

    if not pool_df.empty:
        for row_index, row in pool_df.iterrows():
            prompt_idx = int(row['prompt_idx'])
            text_hash = str(row['text_hash'])
            pool_key_to_index[(prompt_idx, text_hash)] = row_index
        current_counts = pool_df.groupby('prompt_idx')['text_hash'].nunique().astype(int).to_dict()

    for _, row in round_df.iterrows():
        prompt_idx = int(row['prompt_idx'])
        text_hash = str(row['text_hash'])
        row_count = _positive_count(row.get('count', 1))
        prompt_stats = stats.setdefault(
            prompt_idx,
            {
                'generated_attempts': 0,
                'unique_accepted': 0,
                'duplicates_discarded': 0,
                'over_target_discarded': 0,
            },
        )
        prompt_stats['generated_attempts'] += row_count

        key = (prompt_idx, text_hash)
        if key in pool_key_to_index:
            pool_row_index = pool_key_to_index[key]
            pool_df.at[pool_row_index, 'count'] = _positive_count(pool_df.at[pool_row_index, 'count']) + row_count
            pool_df.at[pool_row_index, round_count_col] = int(pool_df.at[pool_row_index, round_count_col]) + row_count
            prompt_stats['duplicates_discarded'] += row_count
            continue
        if key in accepted_key_to_pos:
            accepted_pos = accepted_key_to_pos[key]
            accepted_rows[accepted_pos]['count'] = _positive_count(accepted_rows[accepted_pos].get('count', 1)) + row_count
            accepted_rows[accepted_pos][round_count_col] = int(
                accepted_rows[accepted_pos].get(round_count_col, 0)
            ) + row_count
            prompt_stats['duplicates_discarded'] += row_count
            continue
        if int(current_counts.get(prompt_idx, 0)) >= target_unique:
            prompt_stats['over_target_discarded'] += row_count
            continue

        row = row.copy()
        row['count'] = row_count
        row[round_count_col] = row_count
        accepted_key_to_pos[key] = len(accepted_rows)
        accepted_rows.append(row)
        current_counts[prompt_idx] = int(current_counts.get(prompt_idx, 0)) + 1
        prompt_stats['unique_accepted'] += 1

    if accepted_rows:
        accepted_df = pd.DataFrame(accepted_rows)
        accepted_df = accepted_df[_pool_output_columns(accepted_df)].copy()
        next_pool = pd.concat([pool_df, accepted_df], ignore_index=True)
    else:
        accepted_df = pd.DataFrame(columns=_pool_columns())
        next_pool = pool_df.copy()

    return next_pool, accepted_df, stats


def _materialize_final_unique_samples(
    pool_df,
    output_path: Path,
    target_unique: int,
    skip_prompt_idxs=None,
):
    import pandas as pd

    sorted_df = pool_df.copy()
    skip_prompt_idxs = set(skip_prompt_idxs or [])
    if skip_prompt_idxs:
        sorted_df = sorted_df[~sorted_df['prompt_idx'].isin(skip_prompt_idxs)].copy()

    if sorted_df.empty:
        round_count_cols = _round_count_columns(sorted_df)
        final_df = pd.DataFrame(
            columns=[
                'prompt_idx',
                'sample_idx',
                'behavior_id',
                'category',
                'full_continuation',
                'count',
                *round_count_cols,
            ]
        )
        final_df.to_csv(output_path, index=False)
        return str(output_path)

    sorted_df['_source_round_sort'] = pd.to_numeric(
        sorted_df['source_round'], errors='coerce'
    ).fillna(0)
    sorted_df['_source_sample_idx_sort'] = pd.to_numeric(
        sorted_df['source_sample_idx'], errors='coerce'
    ).fillna(0)
    sorted_df['_pool_order'] = range(len(sorted_df))
    final_df = (
        sorted_df
        .sort_values(['prompt_idx', '_source_round_sort', '_source_sample_idx_sort', '_pool_order'])
        .groupby('prompt_idx')
        .head(target_unique)
        .copy()
    )
    final_df['sample_idx'] = final_df.groupby('prompt_idx').cumcount()
    final_df['count'] = final_df['count'].map(_positive_count)
    _normalize_round_count_columns(final_df)
    round_count_cols = _round_count_columns(final_df)
    output_columns = [
        'prompt_idx',
        'sample_idx',
        'behavior_id',
        'category',
        'full_continuation',
        'count',
        *round_count_cols,
    ]
    final_df = final_df[output_columns].copy()
    final_df.to_csv(output_path, index=False)
    return str(output_path)


def run_unique_random_sampling(
    model_config: Dict,
    gen_config: Dict,
    output_dir: Path,
    args,
    categories: List[str],
) -> str:
    """Run random sampling until each prompt has n unique continuations."""
    import pandas as pd
    from modules.random_sampler import RandomSampler, load_behaviors

    target_unique = args.n_samples
    output_path = output_dir / "samples.csv"
    pool_path = output_dir / "samples.unique_pool.csv"
    state_path = output_dir / "unique_state.json"
    rounds_dir = output_dir / "unique_rounds"
    tmp_round_path = output_dir / "round_tmp.csv"

    behaviors = load_behaviors(
        args.behaviors_csv,
        categories=categories,
        start_idx=args.start_idx,
        max_count=args.max_count,
    )
    if not behaviors:
        print("[UNIQUE] No behaviors selected")
        return str(output_path)

    skip_behavior_ids = _load_behavior_ids_file(args.unique_skip_behavior_ids_file)
    skip_prompt_idxs = {
        prompt_idx
        for prompt_idx, behavior in enumerate(behaviors)
        if behavior.get('BehaviorID', '') in skip_behavior_ids
    }
    if skip_behavior_ids:
        matched_behavior_ids = {
            behavior.get('BehaviorID', '')
            for behavior in behaviors
            if behavior.get('BehaviorID', '') in skip_behavior_ids
        }
        missing_behavior_ids = sorted(skip_behavior_ids - matched_behavior_ids)
        print(
            f"[UNIQUE] Skipping {len(skip_prompt_idxs)} prompts from "
            f"{args.unique_skip_behavior_ids_file}"
        )
        if missing_behavior_ids:
            print(
                f"[UNIQUE] Skip file IDs not found in selected behaviors: "
                f"{missing_behavior_ids[:20]}"
            )

    state = _init_unique_state(state_path, behaviors, target_unique, args)
    if skip_prompt_idxs:
        state['skipped_prompt_count'] = len(skip_prompt_idxs)
        state['skipped_prompt_idxs'] = sorted(skip_prompt_idxs)
        state['skipped_behavior_ids'] = sorted(skip_behavior_ids)
        for prompt_idx in skip_prompt_idxs:
            entry = _ensure_state_prompt(state, prompt_idx, behaviors[prompt_idx], target_unique)
            entry['skipped'] = True

    _import_existing_samples_to_pool(output_path, pool_path, state, behaviors, target_unique)
    pool_df = _load_unique_pool(pool_path)
    _update_current_unique_counts(state, pool_df, behaviors, target_unique)
    _write_unique_state(state_path, state)

    failed_prompt_idxs = _get_failed_prompt_idxs(
        pool_df,
        len(behaviors),
        target_unique,
        skip_prompt_idxs=skip_prompt_idxs,
    )
    if not failed_prompt_idxs:
        print(
            f"[UNIQUE] All non-skipped prompts already have "
            f"{target_unique} unique samples"
        )
        return _materialize_final_unique_samples(
            pool_df,
            output_path,
            target_unique,
            skip_prompt_idxs=skip_prompt_idxs,
        )

    temperature = args.temperature or gen_config.get('temperature', 1.0)
    top_p = args.top_p or gen_config.get('top_p', 0.9)
    top_k = args.k if args.k is not None else gen_config.get('k', 5)
    min_p = args.min_p if args.min_p is not None else gen_config.get('min_p', 0.0)
    max_tokens = args.max_tokens or gen_config.get('max_tokens', 512)
    base_seed = args.unique_base_seed if args.unique_base_seed is not None else (
        args.seed if args.seed is not None else 42
    )

    sampler = RandomSampler(
        model_path=model_config['model_path'],
        model_type=model_config['model_type'],
        dtype=str(model_config.get('dtype', 'float16')),
        num_gpus=int(model_config.get('num_gpus', 2)),
        max_model_len=int(model_config.get('max_model_len', 2048)),
        gpu_memory_utilization=float(model_config.get('gpu_memory_utilization', 0.7)),
    )

    stall_rounds = 0
    sampler.initialize()
    try:
        while int(state.get('rounds_completed', 0)) < args.unique_max_rounds:
            pool_df = _load_unique_pool(pool_path)
            failed_prompt_idxs = _get_failed_prompt_idxs(
                pool_df,
                len(behaviors),
                target_unique,
                skip_prompt_idxs=skip_prompt_idxs,
            )
            if not failed_prompt_idxs:
                break

            max_deficit = _get_max_deficit(
                pool_df,
                len(behaviors),
                target_unique,
                skip_prompt_idxs=skip_prompt_idxs,
            )
            retry_samples = args.unique_retry_samples or target_unique
            retry_samples = max(1, int(retry_samples))
            round_idx = int(state.get('rounds_completed', 0))
            round_seed = int(base_seed + round_idx)

            subset_behaviors = [behaviors[prompt_idx] for prompt_idx in failed_prompt_idxs]
            local_to_original = {
                local_idx: prompt_idx
                for local_idx, prompt_idx in enumerate(failed_prompt_idxs)
            }

            if args.keep_round_files:
                rounds_dir.mkdir(parents=True, exist_ok=True)
                round_output_path = rounds_dir / f"round_{round_idx:04d}_seed_{round_seed}.csv"
            else:
                round_output_path = tmp_round_path
                if round_output_path.exists():
                    round_output_path.unlink()

            print(
                f"[UNIQUE] Round {round_idx}: seed={round_seed}, "
                f"prompts={len(failed_prompt_idxs)}, retry_samples={retry_samples}, "
                f"max_deficit={max_deficit}"
            )

            round_df = sampler.generate_samples(
                behaviors=subset_behaviors,
                output_path=str(round_output_path),
                n_samples=retry_samples,
                max_tokens=max_tokens,
                temperature=temperature,
                top_p=top_p,
                top_k=top_k,
                min_p=min_p,
                seed=round_seed,
                resume=False,
            )
            if round_df is None or round_df.empty:
                round_df = pd.read_csv(round_output_path, low_memory=False)

            round_df = _prepare_round_df(round_df, local_to_original, round_idx, round_seed)
            pool_df, accepted_df, round_stats = _merge_round_into_pool(
                pool_df,
                round_df,
                target_unique,
                round_idx,
            )
            pool_df['sample_idx'] = pool_df.groupby('prompt_idx').cumcount()
            _write_unique_pool(pool_path, pool_df.copy())

            if not args.keep_round_files and round_output_path.exists():
                round_output_path.unlink()

            generated_attempts = int(len(round_df))
            unique_accepted = int(len(accepted_df))
            duplicates_discarded = int(
                sum(s['duplicates_discarded'] for s in round_stats.values())
            )
            over_target_discarded = int(
                sum(s['over_target_discarded'] for s in round_stats.values())
            )

            state['total_generated_attempts'] = int(state.get('total_generated_attempts', 0)) + generated_attempts
            state['total_unique_accepted'] = int(state.get('total_unique_accepted', 0)) + unique_accepted
            state['total_duplicates_discarded'] = int(state.get('total_duplicates_discarded', 0)) + duplicates_discarded
            state['total_over_target_discarded'] = int(state.get('total_over_target_discarded', 0)) + over_target_discarded

            for prompt_idx in failed_prompt_idxs:
                behavior = behaviors[prompt_idx]
                entry = _ensure_state_prompt(state, prompt_idx, behavior, target_unique)
                _record_count(entry, 'rounds_attempted', 1)
                entry['last_seed'] = round_seed

            for prompt_idx, prompt_stats in round_stats.items():
                behavior = behaviors[prompt_idx]
                entry = _ensure_state_prompt(state, prompt_idx, behavior, target_unique)
                for key, amount in prompt_stats.items():
                    _record_count(entry, key, amount)

            _update_current_unique_counts(state, pool_df, behaviors, target_unique)
            remaining_prompt_count = len(_get_failed_prompt_idxs(
                pool_df,
                len(behaviors),
                target_unique,
                skip_prompt_idxs=skip_prompt_idxs,
            ))
            state.setdefault('rounds', []).append({
                'round_idx': round_idx,
                'seed': round_seed,
                'prompt_count': len(failed_prompt_idxs),
                'samples_per_prompt_requested': retry_samples,
                'generated_attempts': generated_attempts,
                'unique_accepted': unique_accepted,
                'duplicates_discarded': duplicates_discarded,
                'over_target_discarded': over_target_discarded,
                'remaining_prompt_count': remaining_prompt_count,
            })
            state['rounds_completed'] = round_idx + 1
            _write_unique_state(state_path, state)

            print(
                f"[UNIQUE] Round {round_idx} accepted {unique_accepted}/{generated_attempts} "
                f"new unique rows; remaining prompts={remaining_prompt_count}"
            )

            if remaining_prompt_count == 0:
                break
            if unique_accepted == 0:
                stall_rounds += 1
            else:
                stall_rounds = 0
            if stall_rounds >= args.unique_stall_rounds:
                print(
                    f"[UNIQUE] Stopped after {stall_rounds} stalled rounds "
                    f"with {remaining_prompt_count} prompts still incomplete"
                )
                break
    finally:
        sampler.cleanup()

    pool_df = _load_unique_pool(pool_path)
    _update_current_unique_counts(state, pool_df, behaviors, target_unique)
    failed_prompt_idxs = _get_failed_prompt_idxs(
        pool_df,
        len(behaviors),
        target_unique,
        skip_prompt_idxs=skip_prompt_idxs,
    )
    _write_unique_state(state_path, state)

    if failed_prompt_idxs:
        print("[UNIQUE] Failed to reach target unique samples for all prompts")
        print(f"[UNIQUE] Incomplete prompt count: {len(failed_prompt_idxs)}")
        for prompt_idx in failed_prompt_idxs[:20]:
            entry = state['per_prompt'][str(prompt_idx)]
            print(
                f"  prompt_idx={prompt_idx} behavior_id={entry['behavior_id']} "
                f"unique={entry['current_unique']}/{target_unique} "
                f"generated_attempts={entry['generated_attempts']}"
            )
        if len(failed_prompt_idxs) > 20:
            print(f"  ... {len(failed_prompt_idxs) - 20} more")
        if not args.unique_allow_partial:
            sys.exit(1)

        state['partial_completion'] = True
        state['incomplete_prompt_count'] = len(failed_prompt_idxs)
        state['incomplete_prompt_idxs'] = failed_prompt_idxs
        _write_unique_state(state_path, state)
        print(
            f"[UNIQUE] --unique-allow-partial set; writing partial unique samples "
            f"to {output_path} and continuing"
        )
        return _materialize_final_unique_samples(
            pool_df,
            output_path,
            target_unique,
            skip_prompt_idxs=skip_prompt_idxs,
        )

    print(f"[UNIQUE] Writing final unique samples to {output_path}")
    return _materialize_final_unique_samples(
        pool_df,
        output_path,
        target_unique,
        skip_prompt_idxs=skip_prompt_idxs,
    )


def run_classifier(
    input_path: str,
    output_path: str,
    classifier_config: Dict,
    args,
    categories: List[str],
):
    """Run classifier on samples."""
    script_dir = Path(__file__).parent
    from modules.classifier_preprocess import preprocess_random_classifier_input

    safe_input_path = preprocess_random_classifier_input(
        input_path=input_path,
        output_path=str(Path(input_path).with_suffix('.classifier_safe.csv')),
        behaviors_csv=args.behaviors_csv,
        classifier_model_path=classifier_config['model_path'],
        categories=categories,
        start_idx=args.start_idx,
        max_count=args.max_count,
        max_prompt_tokens=2000,
    )
    classifier_input_path = dedupe_classifier_input(
        input_path=safe_input_path,
        output_path=str(Path(safe_input_path).with_suffix('.classifier_dedup.csv')),
    )
    classifier_output_path = str(Path(output_path).with_suffix('.classifier_dedup.csv'))

    cmd = [
        sys.executable, str(script_dir / 'modules' / 'classifier.py'),
        '--input', classifier_input_path,
        '--output', classifier_output_path,
        '--model-path', classifier_config['model_path'],
        '--num-gpus', str(classifier_config.get('num_gpus', 2)),
        '--max-model-len', str(classifier_config.get('max_model_len', 2048)),
        '--gpu-memory-utilization', str(classifier_config.get('gpu_memory_utilization', 0.7)),
        '--behaviors-csv', args.behaviors_csv,
        '--generation-col', 'full_continuation',
        '--batch-size', str(args.batch_size or 2000),
    ]
    if args.resume:
        cmd.append('--resume')

    run_command(cmd, "Classification")
    expand_classifier_results(
        original_input_path=safe_input_path,
        dedup_classified_path=classifier_output_path,
        output_path=output_path,
    )


def dedupe_classifier_input(input_path: str, output_path: str) -> str:
    """Collapse duplicate classifier inputs and store their multiplicity in count."""
    import pandas as pd

    if not Path(input_path).exists():
        return input_path

    df = pd.read_csv(input_path, low_memory=False)
    if df.empty or 'prompt_idx' not in df.columns or 'full_continuation' not in df.columns:
        return input_path

    df = df.copy()
    if 'count' in df.columns:
        df['count'] = df['count'].map(_positive_count)
    else:
        df['count'] = 1
    round_count_cols = _round_count_columns(df)
    _normalize_round_count_columns(df)

    df['_normalized_text'] = df['full_continuation'].map(_normalize_text)
    df['_text_hash'] = df['_normalized_text'].map(_hash_text)
    df['classifier_key'] = df['prompt_idx'].astype(str) + ':' + df['_text_hash']
    df['_dedup_count'] = df.groupby('classifier_key')['count'].transform('sum')
    for col in round_count_cols:
        df[f'_dedup_{col}'] = df.groupby('classifier_key')[col].transform('sum')

    dedup_df = df.drop_duplicates('classifier_key', keep='first').copy()
    dedup_df['count'] = dedup_df['_dedup_count'].map(_positive_count)
    for col in round_count_cols:
        dedup_df[col] = dedup_df[f'_dedup_{col}'].fillna(0).astype(int)
    dedup_df = dedup_df.drop(
        columns=[
            '_normalized_text',
            '_text_hash',
            '_dedup_count',
            *[f'_dedup_{col}' for col in round_count_cols],
        ]
    )
    if 'sample_idx' in dedup_df.columns:
        dedup_df['sample_idx'] = dedup_df.groupby('prompt_idx').cumcount()

    dedup_df.to_csv(output_path, index=False)
    duplicate_count = int(df['count'].sum() - len(dedup_df))
    print(
        f"[classifier_dedup] {len(df)} rows -> {len(dedup_df)} unique rows "
        f"({duplicate_count} duplicates counted) -> {output_path}"
    )
    return output_path


def expand_classifier_results(
    original_input_path: str,
    dedup_classified_path: str,
    output_path: str,
):
    """Restore classifier results to the original row order."""
    import pandas as pd

    original_df = pd.read_csv(original_input_path, low_memory=False)
    classified_df = pd.read_csv(dedup_classified_path, low_memory=False)
    if (
        original_df.empty
        or 'prompt_idx' not in original_df.columns
        or 'full_continuation' not in original_df.columns
        or 'classifier_key' not in classified_df.columns
        or 'classifier_result' not in classified_df.columns
    ):
        classified_df.to_csv(output_path, index=False)
        return

    original_df = original_df.copy()
    original_df['_normalized_text'] = original_df['full_continuation'].map(_normalize_text)
    original_df['_text_hash'] = original_df['_normalized_text'].map(_hash_text)
    original_df['classifier_key'] = original_df['prompt_idx'].astype(str) + ':' + original_df['_text_hash']

    result_by_key = (
        classified_df
        .drop_duplicates('classifier_key', keep='first')
        .set_index('classifier_key')['classifier_result']
    )
    original_df['classifier_result'] = original_df['classifier_key'].map(result_by_key)
    missing_count = int(original_df['classifier_result'].isna().sum())
    if missing_count:
        raise ValueError(f"Missing classifier results for {missing_count} original rows")

    drop_cols = [
        c for c in [
            'prompt',
            'prefix_used',
            'model_path',
            'full_continuation',
            '_normalized_text',
            '_text_hash',
            'classifier_key',
        ]
        if c in original_df.columns
    ]
    original_df.drop(columns=drop_cols).to_csv(output_path, index=False)
    print(
        f"[classifier_dedup] Expanded {len(classified_df)} classified rows "
        f"back to {len(original_df)} original rows -> {output_path}"
    )


def generate_summary(output_dir: Path, args, config: Dict, categories: List[str]):
    """Generate experiment summary."""
    import pandas as pd

    summary = {
        'experiment': {
            'type': 'random_sampling',
            'model_name': args.model_name,
            'timestamp': datetime.now().isoformat(),
            'categories': categories,
            'n_samples': args.n_samples,
        },
        'config': {
            'model': config.get(args.model_name, {}),
            'generation': config.get('generation', {})
        },
        'results': {}
    }

    classified_path = output_dir / (
        "classified.unique.csv" if getattr(args, 'ensure_unique', False) else "classified.csv"
    )
    if classified_path.exists():
        df = pd.read_csv(classified_path, low_memory=False)
        if 'count' in df.columns:
            sample_counts = df['count'].map(_positive_count)
        else:
            sample_counts = pd.Series(1, index=df.index)
        yes_mask = df['classifier_result'].map(lambda r: str(r).lower() == 'yes')
        yes_count = int(sample_counts[yes_mask].sum())
        total = int(sample_counts.sum())
        n_prompts = df['prompt_idx'].nunique() if 'prompt_idx' in df.columns else total // args.n_samples

        # Per-prompt success (at least one YES)
        if 'prompt_idx' in df.columns:
            prompt_success = df.groupby('prompt_idx')['classifier_result'].apply(
                lambda x: any(str(r).lower() == 'yes' for r in x)
            )
            prompts_with_success = prompt_success.sum()
        else:
            prompts_with_success = "N/A"

        # Convert numpy types to Python native types for JSON serialization
        import numpy as np
        is_numeric = isinstance(prompts_with_success, (int, float, np.integer, np.floating))
        summary['results'] = {
            'total_samples': int(total),
            'n_prompts': int(n_prompts),
            'yes_count': int(yes_count),
            'no_count': int(total - yes_count),
            'sample_success_rate': float(yes_count / total) if total > 0 else 0.0,
            'prompts_with_at_least_one_yes': int(prompts_with_success) if is_numeric else prompts_with_success,
            'prompt_success_rate': float(prompts_with_success / n_prompts) if is_numeric and n_prompts > 0 else "N/A"
        }

    summary_path = output_dir / "summary.json"
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)

    print(f"\nSummary saved to {summary_path}")
    return summary


def main():
    args = parse_args()

    set_cuda_visible_devices(args.gpu_ids)

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
    print(f"N samples per prompt: {args.n_samples}")

    output_dir = setup_output_dir(args.output_dir, args.model_name)
    print(f"Output directory: {output_dir}")

    categories = args.categories or config.get('categories', DEFAULT_CATEGORIES)
    print(f"Categories: {categories}")

    # Run random sampling
    samples_path = str(output_dir / "samples.csv")
    if not args.skip_generation:
        if args.ensure_unique:
            samples_path = run_unique_random_sampling(
                model_config=model_config,
                gen_config=gen_config,
                output_dir=output_dir,
                args=args,
                categories=categories
            )
        else:
            samples_path = run_random_sampling(
                model_config=model_config,
                gen_config=gen_config,
                output_dir=output_dir,
                args=args,
                categories=categories
            )
    else:
        print("\n[SKIP] Generation step")

    # Run classifier
    if not args.skip_classifier and os.path.exists(samples_path):
        classified_name = "classified.unique.csv" if args.ensure_unique else "classified.csv"
        classified_path = str(output_dir / classified_name)
        if args.ensure_unique and os.path.exists(classified_path):
            os.remove(classified_path)
        run_classifier(
            input_path=samples_path,
            output_path=classified_path,
            classifier_config=classifier_config,
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

    if 'total_samples' in summary.get('results', {}):
        r = summary['results']
        print(f"\nResults:")
        print(f"  Total samples: {r['total_samples']}")
        print(f"  Sample success rate: {r['sample_success_rate']*100:.1f}%")
        if r.get('prompt_success_rate') != "N/A":
            print(f"  Prompts with at least 1 success: {r['prompts_with_at_least_one_yes']}/{r['n_prompts']}")
            print(f"  Prompt success rate: {r['prompt_success_rate']*100:.1f}%")


if __name__ == "__main__":
    main()
