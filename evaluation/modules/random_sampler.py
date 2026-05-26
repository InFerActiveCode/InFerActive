"""Random sampling module using vLLM - generates n samples per prompt."""

import os
import sys

if __name__ == "__main__":
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import pandas as pd
from pathlib import Path
from typing import List, Dict, Tuple
from vllm import LLM, SamplingParams
from transformers import AutoTokenizer

try:
    from .model_utils import build_prompt
except ImportError:
    from model_utils import build_prompt


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


def validate_random_sampling_progress(
    behaviors: List[Dict],
    output_csv: str,
    n_samples: int
) -> Tuple[List[Dict], List[Dict], pd.DataFrame]:
    """
    Validate random sampling CSV against behaviors.

    Args:
        behaviors: List of behavior dicts
        output_csv: Path to output CSV
        n_samples: Expected samples per behavior

    Returns:
        Tuple of (completed_behaviors, incomplete_behaviors, valid_existing_df)
    """
    if not os.path.exists(output_csv):
        return [], behaviors, pd.DataFrame()

    df = pd.read_csv(output_csv)

    # Count samples per prompt_idx
    actual_counts = df.groupby('prompt_idx').size().to_dict()

    completed = []
    incomplete = []
    completed_indices = set()

    for idx, b in enumerate(behaviors):
        actual = actual_counts.get(idx, 0)
        if actual == n_samples:
            completed.append(b)
            completed_indices.add(idx)
        else:
            incomplete.append(b)

    # Keep only completed rows
    if completed_indices:
        valid_df = df[df['prompt_idx'].isin(completed_indices)].copy()
    else:
        valid_df = pd.DataFrame()

    return completed, incomplete, valid_df


class RandomSampler:
    """Generate n random samples per prompt using vLLM."""

    def __init__(
        self,
        model_path: str,
        model_type: str = "llama2",
        dtype: str = "float16",
        num_gpus: int = 2,
        max_model_len: int = 2048,
        gpu_memory_utilization: float = 0.7
    ):
        self.model_path = model_path
        self.model_type = model_type
        self.dtype = dtype
        self.num_gpus = num_gpus
        self.max_model_len = max_model_len
        self.gpu_memory_utilization = gpu_memory_utilization

        self.llm = None
        self.tokenizer = None
        self.terminator_token_ids: List[int] = []

    def initialize(self):
        """Load model and tokenizer."""
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

        print("Model loaded!")

    def cleanup(self):
        """Release GPU memory."""
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

    def generate_samples(
        self,
        behaviors: List[Dict],
        output_path: str,
        n_samples: int = 10,
        max_tokens: int = 512,
        temperature: float = 1.0,
        top_p: float = 0.9,
        top_k: int = -1,
        min_p: float = 0.0,
        seed: int = None,
        resume: bool = False
    ) -> pd.DataFrame:
        """
        Generate n random samples per prompt, saving after each prompt.

        Args:
            behaviors: List of behavior dicts with 'Behavior', 'ContextString', etc.
            output_path: Output CSV path
            n_samples: Number of samples per prompt
            max_tokens: Max tokens to generate
            temperature: Sampling temperature
            top_p: Top-p sampling
            top_k: Top-k sampling (-1 disables top-k filtering)
            seed: Random seed
            resume: Resume from existing output (auto-detected)

        Returns:
            DataFrame with all samples
        """
        # Always validate existing progress from output_path
        completed, incomplete, existing_df = validate_random_sampling_progress(
            behaviors, output_path, n_samples
        )

        if not incomplete:
            print(f"All {len(completed)} behaviors already completed!")
            return pd.read_csv(output_path)

        print(f"[INFO] Completed: {len(completed)}, Processing: {len(incomplete)}")

        # Build index mapping for incomplete behaviors
        remaining_behaviors = incomplete
        original_indices = []
        for idx, b in enumerate(behaviors):
            if b in incomplete:
                original_indices.append(idx)

        # Save valid data to output so we can append incrementally
        if not existing_df.empty:
            existing_df.to_csv(output_path, index=False)
            print(f"[INFO] Saved {len(existing_df)} valid rows")

        total_prompts = len(remaining_behaviors)
        if total_prompts == 0:
            print("No behaviors to process!")
            return existing_df if not existing_df.empty else pd.DataFrame()

        print(f"Generating {n_samples} samples for {total_prompts} prompts...")
        print(f"Total samples to generate: {total_prompts * n_samples}")

        # Sampling params
        effective_top_k = top_k if top_k > 0 else -1
        sampling_params = SamplingParams(
            n=n_samples,
            temperature=temperature,
            top_p=top_p,
            top_k=effective_top_k,
            min_p=min_p,
            max_tokens=max_tokens,
            seed=seed,
            stop_token_ids=self.terminator_token_ids,
        )
        print(
            f"Sampling: n={n_samples}, temperature={temperature}, top_p={top_p}, "
            f"top_k={effective_top_k}, min_p={min_p}, seed={seed}"
        )

        # Determine if output file already has content (existing_df was written above)
        file_initialized = not existing_df.empty

        for prompt_idx_local, (b, orig_idx) in enumerate(zip(remaining_behaviors, original_indices)):
            behavior = b.get('Behavior', '')
            context = b.get('ContextString', '')

            if context and context.strip():
                prompt_text = f"{context}\n\n---\n\n{behavior}"
            else:
                prompt_text = behavior

            prompt_token_ids = self.tokenizer.encode(
                build_prompt(prompt_text, self.model_type, self.tokenizer),
                add_special_tokens=True,
            )

            # Use the same tokenized prompt path as tree exploration.
            outputs = self.llm.generate(
                [{"prompt_token_ids": prompt_token_ids}],
                sampling_params,
            )

            new_rows = [
                {
                    'prompt_idx': orig_idx,
                    'sample_idx': sample_idx,
                    'behavior_id': b.get('BehaviorID', f'behavior_{orig_idx}'),
                    'category': b.get('FunctionalCategory', 'standard'),
                    'full_continuation': sample_output.text.strip()
                }
                for sample_idx, sample_output in enumerate(outputs[0].outputs)
            ]

            # Append only new rows — O(n_samples) regardless of total progress
            new_df = pd.DataFrame(new_rows)
            if file_initialized:
                new_df.to_csv(output_path, mode='a', header=False, index=False)
            else:
                new_df.to_csv(output_path, mode='w', header=True, index=False)
                file_initialized = True

            print(f"[{prompt_idx_local+1}/{total_prompts}] prompt {orig_idx} done ({n_samples} samples)")

        # Final sort (once at the end)
        df = pd.read_csv(output_path)
        df = df.sort_values(['prompt_idx', 'sample_idx']).reset_index(drop=True)
        df.to_csv(output_path, index=False)
        print(f"\nResults saved to {output_path}")
        print(f"Total rows: {len(df)}")

        return df


def load_behaviors(csv_path: str, categories: List[str] = None, start_idx: int = 0, max_count: int = None) -> List[Dict]:
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

    # Apply slicing
    if max_count:
        behaviors = behaviors[start_idx:start_idx + max_count]
    else:
        behaviors = behaviors[start_idx:]

    return behaviors


def main():
    import argparse

    parser = argparse.ArgumentParser(description='Random sampling generation')
    parser.add_argument('--behaviors-csv', '-b', required=True, help='Behaviors CSV file')
    parser.add_argument('--output', '-o', required=True, help='Output CSV file')
    parser.add_argument('--model-path', required=True, help='Path to model')
    parser.add_argument('--model-type', default='llama2', help='Model type')
    parser.add_argument('--dtype', default='float16', help='Model dtype')
    parser.add_argument('--num-gpus', type=int, default=2, help='Number of GPUs')
    parser.add_argument('--max-model-len', type=int, default=2048, help='Max model length')
    parser.add_argument('--gpu-memory-utilization', type=float, default=0.7)
    parser.add_argument('--n-samples', '-n', type=int, default=10, help='Samples per prompt')
    parser.add_argument('--max-tokens', type=int, default=512, help='Max tokens')
    parser.add_argument('--temperature', '-t', type=float, default=1.0, help='Temperature')
    parser.add_argument('--top-p', type=float, default=0.9, help='Top-p sampling')
    parser.add_argument('--top-k', type=int, default=-1, help='Top-k sampling (-1 disables top-k filtering)')
    parser.add_argument('--min-p', type=float, default=0.0, help='Min-p sampling')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--resume', '-r', action='store_true')
    parser.add_argument('--categories', nargs='+', default=['contextual', 'standard'])
    parser.add_argument('--start-idx', type=int, default=0)
    parser.add_argument('--max-count', type=int, default=None)
    args = parser.parse_args()

    # Load behaviors
    behaviors = load_behaviors(
        args.behaviors_csv,
        categories=args.categories,
        start_idx=args.start_idx,
        max_count=args.max_count
    )
    print(f"Loaded {len(behaviors)} behaviors")

    # Validate existing progress before loading model
    print(f"[INFO] Validating random sampling progress...")
    completed, incomplete, valid_df = validate_random_sampling_progress(
        behaviors, args.output, args.n_samples
    )

    print(f"[INFO] Completed: {len(completed)}, Incomplete: {len(incomplete)}")

    if not incomplete:
        print(f"[INFO] All {len(completed)} behaviors already processed. Skipping model load.")
        return

    # Initialize sampler
    sampler = RandomSampler(
        model_path=args.model_path,
        model_type=args.model_type,
        dtype=args.dtype,
        num_gpus=args.num_gpus,
        max_model_len=args.max_model_len,
        gpu_memory_utilization=args.gpu_memory_utilization
    )
    sampler.initialize()

    # Generate
    sampler.generate_samples(
        behaviors=behaviors,
        output_path=args.output,
        n_samples=args.n_samples,
        max_tokens=args.max_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
        top_k=args.top_k,
        min_p=args.min_p,
        seed=args.seed,
        resume=args.resume
    )

    sampler.cleanup()


if __name__ == "__main__":
    main()
