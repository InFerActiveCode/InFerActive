"""Continuation generation module using vLLM."""

import os
import sys

# Add module directory to path for direct execution
if __name__ == "__main__":
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import json
import pandas as pd
from pathlib import Path
from typing import List, Dict, Any, Tuple
from vllm import LLM, SamplingParams
from transformers import AutoTokenizer

DEFAULT_CUDA_VISIBLE_DEVICES = "2,3,4,5"
if "CUDA_VISIBLE_DEVICES" not in os.environ:
    os.environ["CUDA_VISIBLE_DEVICES"] = DEFAULT_CUDA_VISIBLE_DEVICES

# Support both package import and direct execution
try:
    from .model_utils import get_template, PROMPT_TEMPLATES
except ImportError:
    from model_utils import get_template, PROMPT_TEMPLATES


def _get_terminator_token_ids(tokenizer: AutoTokenizer, model_type: str) -> List[int]:
    """Return model terminator token ids used to end assistant generation."""
    terminator_ids: List[int] = []
    eos_token_id = getattr(tokenizer, "eos_token_id", None)

    if isinstance(eos_token_id, int):
        terminator_ids.append(eos_token_id)
    elif isinstance(eos_token_id, (list, tuple)):
        terminator_ids.extend(int(token_id) for token_id in eos_token_id)

    # Meta Llama 3.1 Instruct config uses eos_token_id=[128001, 128008, 128009].
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


def _read_csv_preserving_empty_strings(csv_path: str) -> pd.DataFrame:
    """Read project CSVs without coercing blank text fields to NaN."""
    return pd.read_csv(csv_path, keep_default_na=False)


def _coerce_text(value: Any) -> str:
    """Normalize CSV/model values into text for continuation assembly."""
    if pd.isna(value):
        return ""
    if isinstance(value, str):
        return value
    return str(value)


def _split_terminated_leaf_rows(
    leaf_rows: List[Dict[str, Any]],
    terminator_token_ids: set[int],
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """Split leaf rows into rows needing continuation and already-terminated rows."""
    terminated_rows = [
        row for row in leaf_rows
        if row.get('leaf_token_id') in terminator_token_ids
    ]
    continuation_rows = [
        row for row in leaf_rows
        if row.get('leaf_token_id') not in terminator_token_ids
    ]
    return continuation_rows, terminated_rows


def collect_all_paths_with_text(
    node: Dict[str, Any],
    current_tokens: List[int] = None,
    current_texts: List[str] = None,
    current_prob: float = 1.0,
    current_is_greedy: List[bool] = None
) -> List[Tuple[List[int], List[str], str, float, List[bool]]]:
    """
    Collect all leaf paths from a tree.

    Returns:
        List of (token_ids, texts, leaf_id, cumulative_prob, is_greedy_list, leaf_token_id) tuples
    """
    if current_tokens is None:
        current_tokens = []
    if current_texts is None:
        current_texts = []
    if current_is_greedy is None:
        current_is_greedy = []

    # Add current node's token (except root)
    if node.get("id") != "root":
        current_tokens = current_tokens + [node.get("token_id")]
        current_texts = current_texts + [node.get("text", "")]
        current_prob = current_prob * node.get("prob", 1.0)

    children = node.get("children", [])

    # Leaf node
    if not children:
        return [(current_tokens, current_texts, node.get("id", ""), current_prob, current_is_greedy, node.get("token_id"))]

    # Find max prob for greedy tracking
    max_prob = max(child.get("prob", 0.0) for child in children)

    all_paths = []
    for child in children:
        child_prob = child.get("prob", 0.0)
        is_greedy = (child_prob >= max_prob)
        child_is_greedy = current_is_greedy + [is_greedy]
        child_paths = collect_all_paths_with_text(
            child, current_tokens, current_texts, current_prob, child_is_greedy
        )
        all_paths.extend(child_paths)

    return all_paths


def calculate_greedy_from_depth(is_greedy_list: List[bool]) -> int:
    """Calculate from which depth the path becomes consistently greedy (1-indexed)."""
    if not is_greedy_list:
        return 1

    greedy_from = len(is_greedy_list)
    for i in range(len(is_greedy_list) - 1, -1, -1):
        if is_greedy_list[i]:
            greedy_from = i
        else:
            break

    return greedy_from + 1


def process_tree_files(dir_path: str) -> List[Dict[str, Any]]:
    """
    Process all JSON tree files and extract leaf paths.

    Args:
        dir_path: Directory containing JSON tree files

    Returns:
        List of path dictionaries
    """
    dir_path = Path(dir_path)
    all_results = []

    if dir_path.is_file() and dir_path.suffix == '.json':
        json_files = [dir_path]
        dir_path = dir_path.parent
    else:
        json_files = list(dir_path.glob("**/*.json"))
        json_files = [f for f in json_files if f.name != "batch_summary.json"]

    print(f"Found {len(json_files)} JSON files...")

    for json_file in json_files:
        try:
            with open(json_file, 'r', encoding='utf-8') as f:
                data = json.load(f)

            metadata = data.get("metadata", {})
            prompt = metadata.get("prompt", "")
            model_path = metadata.get("model_path", "")
            generated_at = metadata.get("generated_at", "")

            stats = metadata.get("statistics", {})
            total_nodes = stats.get("total_nodes", 0)
            leaf_nodes_count = stats.get("leaf_nodes", 0)
            max_depth = stats.get("max_depth", 0)

            tree = data.get("tree", {})
            leaf_paths = collect_all_paths_with_text(tree)

            for token_ids, texts, leaf_id, path_prob, is_greedy_list, leaf_token_id in leaf_paths:
                greedy_from_depth = calculate_greedy_from_depth(is_greedy_list)
                all_results.append({
                    "file_name": json_file.name,
                    "prompt": prompt,
                    "response": "".join(texts),
                    "leaf_id": leaf_id,
                    "depth": len(texts),
                    "total_nodes": total_nodes,
                    "leaf_nodes": leaf_nodes_count,
                    "max_depth": max_depth,
                    "model_path": model_path,
                    "generated_at": generated_at,
                    "token_ids": token_ids,
                    "texts": texts,
                    "path_probability": path_prob,
                    "is_greedy_list": is_greedy_list,
                    "greedy_from_depth": greedy_from_depth,
                    "leaf_token_id": leaf_token_id,
                })

            print(f"  ✓ {json_file.name}: {len(leaf_paths)} paths")

        except Exception as e:
            print(f"  ✗ {json_file.name}: Error - {e}")

    return all_results


class ContinuationGenerator:
    """Generate continuations for tree leaf paths using vLLM."""

    def __init__(
        self,
        model_path: str,
        model_type: str = "llama2",
        dtype: str = "float16",
        num_gpus: int = 2,
        max_model_len: int = 2048,
        gpu_memory_utilization: float = 0.7
    ):
        """
        Initialize the continuation generator.

        Args:
            model_path: Path to the model
            model_type: Model type (llama2, llama3, etc.)
            num_gpus: Number of GPUs for tensor parallelism
            max_model_len: Maximum model length
            gpu_memory_utilization: GPU memory utilization
        """
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
        """Load model and tokenizer."""
        print(f"Loading continuation model from {self.model_path}...")

        self.tokenizer = AutoTokenizer.from_pretrained(self.model_path, use_fast=True, trust_remote_code=True)

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
            model_type=self.model_type
        )

        print("Continuation model loaded!")

    def cleanup(self):
        """Release GPU memory."""
        print("Cleaning up continuation generator...")
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

    def generate_continuations(
        self,
        tree_dir: str,
        output_path: str,
        max_tokens: int = 200,
        batch_size: int = 2000,
        checkpoint_every: int = 10000,
        resume: bool = False,
        temperature: float = 0.0,
        top_p: float = 1.0,
        top_k: int = -1,
        seed: int = 42,
        num_samples: int = 1
    ) -> pd.DataFrame:
        """
        Generate continuations for all leaf paths in tree files.

        Args:
            tree_dir: Directory containing tree JSON files
            output_path: Output CSV path
            max_tokens: Maximum tokens to generate
            batch_size: Batch size for processing
            checkpoint_every: Save checkpoint every N rows
            resume: Whether to resume from checkpoint
            temperature: Sampling temperature (0.0 for greedy, >0 for random)
            top_p: Top-p (nucleus) sampling parameter
            top_k: Top-k sampling parameter (-1 disables top-k filtering)
            seed: Random seed for reproducible sampling (only used when temperature > 0)

        Returns:
            DataFrame with continuations
        """
        terminator_token_ids = set(self.terminator_token_ids)
        num_samples = max(1, num_samples)

        # Check for checkpoint first
        checkpoint_file = output_path.replace('.csv', '_checkpoint.csv')
        if resume and os.path.exists(checkpoint_file):
            # Resume from checkpoint - use same validation as non-resume path
            completed, incomplete, existing_df = validate_continuation_progress(tree_dir, output_path, num_samples)

            print(f"Processing trees from {tree_dir}...")
            all_data_full = process_tree_files(tree_dir)
            all_data = [d for d in all_data_full if d['file_name'] in set(incomplete)]

            all_data, eos_data = _split_terminated_leaf_rows(all_data, terminator_token_ids)

            if eos_data:
                print(f"[INFO] {len(eos_data)} terminated leaves (skipping continuation, added directly)")

            total_rows = len(all_data)
            print(f"Total leaf paths: {total_rows}")

            checkpoint_df = _read_csv_preserving_empty_strings(checkpoint_file)
            # Filter checkpoint to only files in our target set
            checkpoint_df = checkpoint_df[checkpoint_df['file_name'].isin(set(incomplete))]
            # Trim incomplete prompt from end
            if len(checkpoint_df) > 0:
                last_file = checkpoint_df['file_name'].iloc[-1]
                expected = sum(num_samples for d in all_data if d['file_name'] == last_file)
                actual = (checkpoint_df['file_name'] == last_file).sum()
                if actual != expected:
                    checkpoint_df = checkpoint_df[checkpoint_df['file_name'] != last_file]
                    print(f"Trimmed incomplete prompt '{last_file}' ({actual}/{expected} rows)")
            start_idx = len(checkpoint_df)
            print(f"Resuming from checkpoint at row {start_idx}")
            continuations = checkpoint_df['continuation'].tolist()
            prefixes = checkpoint_df['prefix_used'].tolist()
        else:
            # Validate and filter to incomplete files only
            completed, incomplete, existing_df = validate_continuation_progress(tree_dir, output_path, num_samples)

            if not incomplete:
                print(f"All files already completed!")
                return _read_csv_preserving_empty_strings(output_path)

            print(f"[INFO] Completed: {len(completed)}, Processing: {len(incomplete)}")

            # Process only incomplete files
            print(f"Processing trees from {tree_dir}...")
            all_data_full = process_tree_files(tree_dir)
            all_data = [d for d in all_data_full if d['file_name'] in incomplete]

            all_data, eos_data = _split_terminated_leaf_rows(all_data, terminator_token_ids)

            if eos_data:
                print(
                    f"[INFO] {len(eos_data)} terminated leaves "
                    f"(skipping continuation, added directly)"
                )

            total_rows = len(all_data)
            print(f"Leaf paths to process: {total_rows}")

            start_idx = 0
            continuations = []
            prefixes = []

        if total_rows == 0 and not eos_data:
            print("No data to process!")
            return existing_df if not existing_df.empty else pd.DataFrame()

        # Build inputs
        if temperature > 0:
            effective_top_k = top_k if top_k > 0 else -1
            if seed is not None:
                seed_msg = (
                    f"seed={seed}"
                    if num_samples == 1
                    else f"seeds={seed}..{seed + num_samples - 1} per leaf"
                )
                print(
                    f"Using random sampling: temperature={temperature}, top_p={top_p}, "
                    f"top_k={effective_top_k}, {seed_msg}"
                )
            else:
                print(
                    f"Using random sampling: temperature={temperature}, top_p={top_p}, "
                    f"top_k={effective_top_k}"
                )
        else:
            print("Using greedy decoding (temperature=0.0)")

        inputs = []
        prefix_texts = []

        for item in all_data:
            prompt = item['prompt']
            token_ids = item['token_ids']
            texts = item['texts']

            # Build base prompt
            base_prompt = self.template['prompt'].format(instruction=prompt)
            base_tokens = self.tokenizer.encode(base_prompt, add_special_tokens=True)

            # Full token sequence
            full_token_ids = base_tokens + token_ids
            inputs.append(full_token_ids)
            prefix_texts.append("".join(texts))

        # Output mapping rows (leaf x sample) for checkpoint/final merge
        output_data = []
        for item in all_data:
            for s_idx in range(num_samples):
                mapped_item = item.copy()
                mapped_item['sample_idx'] = s_idx
                output_data.append(mapped_item)
        total_output_rows = len(output_data)

        if start_idx > total_output_rows:
            print(
                f"[WARN] Checkpoint row {start_idx} exceeds expected rows "
                f"{total_output_rows}; trimming checkpoint"
            )
            start_idx = total_output_rows
            continuations = continuations[:start_idx]
            prefixes = prefixes[:start_idx]

        # Batch processing
        print(f"Processing {total_output_rows - start_idx} remaining rows...")
        total_batches = (total_output_rows - start_idx + batch_size - 1) // batch_size

        if total_output_rows > 0:
            for batch_idx, i in enumerate(range(start_idx, total_output_rows, batch_size)):
                batch_end = min(i + batch_size, total_output_rows)
                batch_items = output_data[i:batch_end]

                prompts = []
                batch_prefixes = []
                sp_list = []
                for row_idx, item in enumerate(batch_items, start=i):
                    leaf_idx = row_idx // num_samples
                    sample_seed = (
                        seed + int(item['sample_idx'])
                        if temperature > 0 and seed is not None else None
                    )
                    prompts.append({"prompt_token_ids": inputs[leaf_idx]})
                    batch_prefixes.append(prefix_texts[leaf_idx])
                    # 프롬프트 이후 기준 max_tokens 적용 (max_tokens - tree_depth)
                    sp_list.append(SamplingParams(
                        temperature=temperature, top_p=top_p,
                        top_k=top_k if temperature > 0 and top_k > 0 else -1,
                        max_tokens=max(1, max_tokens - item['depth']),
                        seed=sample_seed,
                        n=1,
                        stop_token_ids=self.terminator_token_ids,
                    ))

                outputs = self.llm.generate(prompts, sampling_params=sp_list)

                for o_idx, o in enumerate(outputs):
                    continuations.append(o.outputs[0].text)
                    prefixes.append(batch_prefixes[o_idx])

                print(f"[{batch_idx+1}/{total_batches}] {len(continuations)}/{total_output_rows} done")

                # Checkpoint
                if len(continuations) % checkpoint_every < batch_size:
                    self._save_checkpoint(output_data, continuations, prefixes, checkpoint_file)
                    print(f">>> Checkpoint saved at row {len(continuations)}")

        # Final save - merge with existing completed data
        final_data = self._build_final_data(output_data, continuations, prefixes)

        # Add EOS leaf nodes directly (no continuation generated)
        if eos_data:
            for item in eos_data:
                for s_idx in range(num_samples):
                    final_data.append({
                        'file_name': item['file_name'],
                        'response': item['response'],
                        'leaf_id': item['leaf_id'],
                        'depth': item['depth'],
                        'total_nodes': item['total_nodes'],
                        'leaf_nodes': item['leaf_nodes'],
                        'max_depth': item['max_depth'],
                        'model_path': item['model_path'],
                        'generated_at': item['generated_at'],
                        'path_probability': item['path_probability'],
                        'greedy_from_depth': item['greedy_from_depth'],
                        'sample_idx': s_idx,
                        'prefix_used': "".join(item['texts']),
                        'continuation': '',
                        'full_continuation': "".join(item['texts']),
                        'is_eos': True,
                    })

        df = pd.DataFrame(final_data)

        if not existing_df.empty:
            df = pd.concat([existing_df, df], ignore_index=True)

        df.to_csv(output_path, index=False)
        print(f"\nResults saved to {output_path} ({len(df)} total rows)")

        # Remove checkpoint
        if os.path.exists(checkpoint_file):
            os.remove(checkpoint_file)

        return df

    def _save_checkpoint(self, all_data, continuations, prefixes, checkpoint_file):
        """Save checkpoint."""
        checkpoint_data = []
        for j, item in enumerate(all_data[:len(continuations)]):
            prefix_text = _coerce_text(prefixes[j])
            continuation_text = _coerce_text(continuations[j])
            checkpoint_data.append({
                'file_name': item['file_name'],
                'response': item['response'],
                'leaf_id': item['leaf_id'],
                'depth': item['depth'],
                'path_probability': item['path_probability'],
                'greedy_from_depth': item['greedy_from_depth'],
                'prefix_used': prefix_text,
                'continuation': continuation_text
            })
        pd.DataFrame(checkpoint_data).to_csv(checkpoint_file, index=False)

    def _build_final_data(self, all_data, continuations, prefixes):
        """Build final data list."""
        final_data = []
        for j, item in enumerate(all_data):
            prefix_text = _coerce_text(prefixes[j])
            continuation_text = _coerce_text(continuations[j])
            final_data.append({
                'file_name': item['file_name'],
                'response': item['response'],
                'leaf_id': item['leaf_id'],
                'depth': item['depth'],
                'total_nodes': item['total_nodes'],
                'leaf_nodes': item['leaf_nodes'],
                'max_depth': item['max_depth'],
                'model_path': item['model_path'],
                'generated_at': item['generated_at'],
                'path_probability': item['path_probability'],
                'greedy_from_depth': item['greedy_from_depth'],
                'sample_idx': item.get('sample_idx', 0),
                'prefix_used': prefix_text,
                'continuation': continuation_text,
                'full_continuation': prefix_text + continuation_text
            })
        return final_data


def generate_greedy_baselines(
    llm,
    tokenizer,
    model_type: str,
    template,
    behaviors: List[Dict],
    output_path: str,
    max_tokens: int = 200
) -> pd.DataFrame:
    """
    Generate greedy baseline completions using already-loaded LLM.

    Args:
        llm: vLLM LLM instance
        template: Prompt template
        behaviors: List of behavior dictionaries
        output_path: Output CSV path
        max_tokens: Max tokens to generate

    Returns:
        DataFrame with greedy results
    """
    print(f"\nGenerating greedy baselines for {len(behaviors)} behaviors...")

    terminator_token_ids = _get_terminator_token_ids(tokenizer, model_type)
    sampling_params = SamplingParams(
        temperature=0.0,
        max_tokens=max_tokens,
        stop_token_ids=terminator_token_ids,
    )

    # Build prompts
    prompts = []
    for b in behaviors:
        behavior = b.get('Behavior', '')
        context = b.get('ContextString', '')

        if context and context.strip():
            prompt_text = f"{context}\n\n---\n\n{behavior}"
        else:
            prompt_text = behavior

        formatted = template['prompt'].format(instruction=prompt_text)
        prompts.append(formatted)

    # Generate
    outputs = llm.generate(prompts, sampling_params)
    generations = [o.outputs[0].text.strip() for o in outputs]

    # Build dataframe
    data = []
    for i, b in enumerate(behaviors):
        data.append({
            'behavior_id': b.get('BehaviorID', f'behavior_{i}'),
            'prompt': b.get('Behavior', ''),
            'context': b.get('ContextString', ''),
            'generation': generations[i],
            'full_continuation': generations[i]  # For classifier compatibility
        })

    df = pd.DataFrame(data)
    df.to_csv(output_path, index=False)
    print(f"Greedy baselines saved to {output_path}")

    return df


def count_leaf_nodes_per_file(tree_dir: str) -> Dict[str, int]:
    """
    Count leaf nodes for each JSON file in tree directory.

    Args:
        tree_dir: Directory containing tree JSON files

    Returns:
        Dict mapping file_name to leaf node count
    """
    tree_dir = Path(tree_dir)
    counts = {}

    if tree_dir.is_file() and tree_dir.suffix == '.json':
        json_files = [tree_dir]
    else:
        json_files = list(tree_dir.glob("**/*.json"))
        json_files = [f for f in json_files if f.name != "batch_summary.json"]

    for json_file in json_files:
        try:
            with open(json_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            stats = data.get("metadata", {}).get("statistics", {})
            counts[json_file.name] = stats.get("leaf_nodes", 0)
        except Exception:
            counts[json_file.name] = -1  # Invalid file

    return counts


def validate_continuation_progress(
    tree_dir: str,
    output_csv: str,
    num_samples: int = 1,
) -> Tuple[List[str], List[str], pd.DataFrame]:
    """
    Validate continuation CSV against tree files.

    Args:
        tree_dir: Directory containing tree JSON files
        output_csv: Path to continuation CSV
        num_samples: Expected number of rows per leaf node

    Returns:
        Tuple of (completed_files, incomplete_files, valid_existing_df)
        - completed_files: JSON files fully processed
        - incomplete_files: JSON files needing (re)processing
        - valid_existing_df: DataFrame with only completed file rows
    """
    expected_counts = count_leaf_nodes_per_file(tree_dir)
    num_samples = max(1, num_samples)

    if not os.path.exists(output_csv):
        return [], list(expected_counts.keys()), pd.DataFrame()

    df = _read_csv_preserving_empty_strings(output_csv)
    actual_counts = df.groupby('file_name').size().to_dict()

    completed = []
    incomplete = []

    for file_name, expected in expected_counts.items():
        expected *= num_samples
        actual = actual_counts.get(file_name, 0)
        if expected > 0 and actual == expected:
            completed.append(file_name)
        else:
            incomplete.append(file_name)

    # Keep only completed file rows
    if completed:
        valid_df = df[df['file_name'].isin(completed)].copy()
    else:
        valid_df = pd.DataFrame()

    return completed, incomplete, valid_df


def load_behaviors(csv_path: str, categories: List[str] = None) -> List[Dict]:
    """Load behaviors from CSV."""
    import csv as csv_module

    if categories is None:
        categories = ['contextual', 'standard']

    behaviors = []
    with open(csv_path, 'r', encoding='utf-8') as f:
        reader = csv_module.DictReader(f)
        for row in reader:
            if row.get('FunctionalCategory', 'standard') in categories:
                behaviors.append({
                    'Behavior': row.get('Behavior', ''),
                    'ContextString': row.get('ContextString', ''),
                    'BehaviorID': row.get('BehaviorID', ''),
                    'SemanticCategory': row.get('SemanticCategory', ''),
                    'FunctionalCategory': row.get('FunctionalCategory', 'standard'),
                })

    return behaviors


def main():
    import argparse

    parser = argparse.ArgumentParser(description='Continuation and greedy baseline generation')
    parser.add_argument('--input', '-i', required=True, help='Input directory with JSON tree files')
    parser.add_argument('--output', '-o', required=True, help='Output CSV file for continuations')
    parser.add_argument('--model-path', required=True, help='Path to model')
    parser.add_argument('--model-type', default='llama2', help='Model type (llama2, llama3)')
    parser.add_argument('--dtype', default='float16', help='Model dtype')
    parser.add_argument('--num-gpus', type=int, default=2, help='Number of GPUs')
    parser.add_argument('--max-model-len', type=int, default=2048, help='Max model length')
    parser.add_argument('--gpu-memory-utilization', type=float, default=0.7, help='GPU memory utilization')
    parser.add_argument('--batch-size', '-b', type=int, default=2000, help='Batch size')
    parser.add_argument('--max-tokens', '-m', type=int, default=200, help='Max tokens to generate')
    parser.add_argument('--checkpoint-every', '-c', type=int, default=10000, help='Checkpoint interval')
    parser.add_argument('--resume', '-r', action='store_true', help='Resume from checkpoint')
    parser.add_argument('--temperature', '-t', type=float, default=1.0, help='Sampling temperature (0.0 for greedy)')
    parser.add_argument('--top-p', type=float, default=1.0, help='Top-p (nucleus) sampling parameter')
    parser.add_argument('--top-k', type=int, default=-1, help='Top-k sampling parameter (-1 disables top-k filtering)')
    parser.add_argument('--seed', '-s', type=int, default=42, help='Random seed for reproducible sampling')
    parser.add_argument('--num-samples', '-n', type=int, default=1, help='Number of samples per leaf (default: 1)')
    # Greedy baseline args
    parser.add_argument('--behaviors-csv', help='Behaviors CSV for greedy baseline')
    parser.add_argument('--greedy-output', help='Output CSV for greedy baselines')
    parser.add_argument('--categories', nargs='+', default=['contextual', 'standard'])
    parser.add_argument('--start-idx', type=int, default=0)
    parser.add_argument('--max-count', type=int, default=None)
    parser.add_argument('--system-prompt-suffix', type=str, default=None,
                        help='Path to file with additional system prompt text')
    args = parser.parse_args()

    # Set system prompt suffix before any model operations
    if args.system_prompt_suffix:
        from model_utils import set_system_prompt_suffix
        with open(args.system_prompt_suffix, 'r', encoding='utf-8') as f:
            suffix_text = f.read().strip()
        set_system_prompt_suffix(suffix_text)

    # Check for checkpoint first (resume takes priority)
    checkpoint_file = args.output.replace('.csv', '_checkpoint.csv')
    if args.resume and os.path.exists(checkpoint_file):
        print(f"[INFO] Checkpoint found: {checkpoint_file}, will resume")
    else:
        # Validate existing progress before loading model
        print(f"[INFO] Validating continuation progress...")
        completed, incomplete, valid_df = validate_continuation_progress(
            args.input,
            args.output,
            args.num_samples,
        )

        print(f"[INFO] Completed: {len(completed)}, Incomplete: {len(incomplete)}")

        if not incomplete:
            print(f"[INFO] All {len(completed)} files already processed. Skipping model load.")
            return

        # Save valid data and prepare for incremental processing
        if not valid_df.empty:
            valid_df.to_csv(args.output, index=False)
            print(f"[INFO] Saved {len(valid_df)} valid rows, will process {len(incomplete)} remaining files")

    cont_gen = ContinuationGenerator(
        model_path=args.model_path,
        model_type=args.model_type,
        dtype=args.dtype,
        num_gpus=args.num_gpus,
        max_model_len=args.max_model_len,
        gpu_memory_utilization=args.gpu_memory_utilization
    )
    cont_gen.initialize()

    # 1. Tree continuation
    cont_gen.generate_continuations(
        tree_dir=args.input,
        output_path=args.output,
        max_tokens=args.max_tokens,
        batch_size=args.batch_size,
        checkpoint_every=args.checkpoint_every,
        resume=args.resume,
        temperature=args.temperature,
        top_p=args.top_p,
        top_k=args.top_k,
        seed=args.seed,
        num_samples=args.num_samples
    )

    # 2. Greedy baseline (if requested)
    if args.behaviors_csv and args.greedy_output:
        behaviors = load_behaviors(args.behaviors_csv, args.categories)
        if args.max_count:
            behaviors = behaviors[args.start_idx:args.start_idx + args.max_count]
        else:
            behaviors = behaviors[args.start_idx:]

        generate_greedy_baselines(
            llm=cont_gen.llm,
            tokenizer=cont_gen.tokenizer,
            model_type=cont_gen.model_type,
            template=cont_gen.template,
            behaviors=behaviors,
            output_path=args.greedy_output,
            max_tokens=args.max_tokens
        )

    cont_gen.cleanup()


if __name__ == "__main__":
    main()
