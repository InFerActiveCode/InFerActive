"""Token-level inference engine with batching for tree exploration."""

import os
import sys

# Add module directory to path for direct execution
if __name__ == "__main__":
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import asyncio
import logging
from typing import Any, Dict, List, Optional, Tuple

import torch
from torch.nn.utils.rnn import pad_sequence
from transformers import AutoModelForCausalLM, AutoTokenizer

# Support both package import and direct execution
try:
    from .token_node import TokenNode
except ImportError:
    from token_node import TokenNode

logger = logging.getLogger(__name__)


class TokenInferenceEngine:
    """
    Token-level inference engine with batched forward passes.
    Used for efficient tree exploration.
    """

    def __init__(
        self,
        model: AutoModelForCausalLM,
        tokenizer: AutoTokenizer,
        model_type: str = "llama2",
        device: str = "cuda:0",
        batch_size: int = 64,
        batch_timeout: float = 0.05
    ):
        """
        Initialize the inference engine.

        Args:
            model: Loaded language model
            tokenizer: Tokenizer for the model
            model_type: Model type (llama2, llama3, qwen, mistral)
            device: Device for inference
            batch_size: Maximum batch size
            batch_timeout: Timeout for batch collection
        """
        self.model = model
        self.tokenizer = tokenizer
        self.model_type = model_type
        self.device = device
        self.batch_size = batch_size
        self.batch_timeout = batch_timeout

        # Batch processing
        self._batch_queue = asyncio.Queue()
        self._broadcast_queue: Optional[asyncio.Queue] = None
        self._worker_task = None

        # KV cache mode (enabled only while active leaves are within budget)
        self._kv_cache_enabled = False
        self._kv_phase_open = False
        self._current_parent_cache: Dict[str, Any] = {}
        self._next_parent_cache: Dict[str, Any] = {}
        self._extra_terminator_token_ids = set()
        if self.model_type == "qwen":
            eos_token_id = getattr(self.tokenizer, "eos_token_id", None)
            if isinstance(eos_token_id, int):
                self._extra_terminator_token_ids.add(eos_token_id)
            endoftext_token_id = self.tokenizer.convert_tokens_to_ids("<|endoftext|>")
            if isinstance(endoftext_token_id, int) and endoftext_token_id >= 0:
                self._extra_terminator_token_ids.add(endoftext_token_id)
        if self.model_type == "gemma":
            eos_token_id = getattr(self.tokenizer, "eos_token_id", None)
            if isinstance(eos_token_id, int):
                self._extra_terminator_token_ids.add(eos_token_id)
            end_of_turn_id = self.tokenizer.convert_tokens_to_ids("<end_of_turn>")
            if isinstance(end_of_turn_id, int) and end_of_turn_id >= 0:
                self._extra_terminator_token_ids.add(end_of_turn_id)

        logger.info(f"TokenInferenceEngine initialized - Device: {device}, Batch: {batch_size}")

    def set_broadcast_queue(self, queue: asyncio.Queue) -> None:
        """Set broadcast queue for real-time updates."""
        self._broadcast_queue = queue

    def start_worker(self):
        """Start the background batch worker."""
        if self._worker_task is None:
            self._worker_task = asyncio.create_task(self._run_batch_worker())

    def stop_worker(self):
        """Stop the background batch worker."""
        if self._worker_task is not None:
            self._worker_task.cancel()
            self._worker_task = None

    def enable_kv_cache_mode(self) -> None:
        """Enable KV cache reuse mode."""
        self._kv_cache_enabled = True

    def disable_kv_cache_mode(self) -> None:
        """Disable KV cache reuse mode and clear cache tensors."""
        self._kv_cache_enabled = False
        self._kv_phase_open = False
        self.clear_kv_cache()

    def begin_kv_phase(self) -> None:
        """Start a phase; expanded nodes are cached for the next phase."""
        if not self._kv_cache_enabled:
            return
        self._kv_phase_open = True
        self._next_parent_cache = {}

    def end_kv_phase(self) -> None:
        """Finish a phase and promote newly cached nodes."""
        if not self._kv_cache_enabled:
            return
        self._current_parent_cache = self._next_parent_cache
        self._next_parent_cache = {}
        self._kv_phase_open = False

    def clear_kv_cache(self) -> None:
        """Drop all cached KV entries."""
        self._current_parent_cache = {}
        self._next_parent_cache = {}

    async def lazy_batched_forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """
        Register a forward pass request for batching.

        Args:
            inputs: Input tensor of token IDs

        Returns:
            Logits for the last token position
        """
        loop = asyncio.get_running_loop()
        future = loop.create_future()

        await self._batch_queue.put((inputs, future))
        return await future

    async def _run_batch_worker(self) -> None:
        """Background worker that batches forward pass requests."""
        try:
            while True:
                batch = []
                loop = asyncio.get_running_loop()
                start_time = loop.time()

                try:
                    item = await self._batch_queue.get()
                    batch.append(item)
                except Exception as e:
                    logger.error(f"Error getting batch item: {e}")
                    continue

                # Collect additional requests
                while len(batch) < self.batch_size:
                    timeout = self.batch_timeout - (loop.time() - start_time)
                    if timeout <= 0:
                        break

                    try:
                        item = await asyncio.wait_for(self._batch_queue.get(), timeout=timeout)
                        batch.append(item)
                    except asyncio.TimeoutError:
                        break
                    except Exception:
                        break

                await self._process_inference_batch(batch)
        except asyncio.CancelledError:
            logger.info("Batch worker cancelled")

    async def _process_inference_batch(self, batch: List[tuple]) -> None:
        """Process a batch of forward pass requests."""
        inputs_list = [req[0] for req in batch]
        futures = [req[1] for req in batch]

        try:
            input_ids_list = []
            orig_lengths = []

            for inp in inputs_list:
                if inp.ndim == 1:
                    inp = inp.unsqueeze(0)
                orig_lengths.append(inp.shape[-1])
                input_ids_list.append(inp.squeeze(0))

            # Pad sequences
            pad_token_id = self.tokenizer.pad_token_id or self.tokenizer.eos_token_id or 0
            padded_input_ids = pad_sequence(
                input_ids_list,
                batch_first=True,
                padding_value=float(pad_token_id)
            )

            # Create attention mask
            attention_mask = (padded_input_ids != pad_token_id).long()

            with torch.no_grad():
                # Get model's input device (for device_map="auto")
                if hasattr(self.model, 'hf_device_map'):
                    input_device = next(iter(self.model.hf_device_map.values()))
                    input_device = f"cuda:{input_device}" if isinstance(input_device, int) else input_device
                else:
                    input_device = self.device

                outputs = self.model(
                    input_ids=padded_input_ids.to(input_device),
                    attention_mask=attention_mask.to(input_device)
                )

            if hasattr(outputs, "logits"):
                logits = outputs.logits

                for i, (future, orig_len) in enumerate(zip(futures, orig_lengths)):
                    if not future.done():
                        last_token_logits = logits[i, orig_len - 1, :].clone()
                        future.set_result(last_token_logits)
                del outputs, logits
            else:
                error = ValueError("Unexpected model output format")
                for future in futures:
                    if not future.done():
                        future.set_exception(error)

        except Exception as e:
            logger.error(f"Batch processing error: {e}", exc_info=True)
            for future in futures:
                if not future.done():
                    future.set_exception(e)

    def _get_input_device(self) -> str:
        """Get model input device for sharded or single-device models."""
        if hasattr(self.model, 'hf_device_map'):
            input_device = next(iter(self.model.hf_device_map.values()))
            return f"cuda:{input_device}" if isinstance(input_device, int) else input_device
        return self.device

    def _store_next_kv_cache(self, node_id: str, past_key_values: Any) -> None:
        """Store node KV for next phase while KV mode is enabled."""
        if not self._kv_cache_enabled or not self._kv_phase_open:
            return
        self._next_parent_cache[node_id] = past_key_values

    def _forward_full_prefix_with_cache(self, prefix: List[int]) -> Tuple[torch.Tensor, Optional[Any]]:
        """Forward full prefix and return next-token logits plus cache."""
        input_device = self._get_input_device()
        input_ids = torch.tensor(prefix, dtype=torch.long, device=input_device).unsqueeze(0)
        attention_mask = torch.ones_like(input_ids, dtype=torch.long, device=input_device)

        with torch.no_grad():
            outputs = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                use_cache=True
            )

        logits = outputs.logits[0, -1, :].clone()
        past_key_values = outputs.past_key_values if hasattr(outputs, "past_key_values") else None
        del outputs
        return logits, past_key_values

    def _forward_from_parent_cache(
        self,
        token_id: int,
        parent_cache: Any
    ) -> Tuple[torch.Tensor, Optional[Any]]:
        """Forward one token using parent KV cache."""
        input_device = self._get_input_device()
        input_ids = torch.tensor([[token_id]], dtype=torch.long, device=input_device)

        # attention_mask length = cached prefix + current token
        try:
            past_len = int(parent_cache[0][0].shape[-2])
        except Exception:
            past_len = 0
        attention_mask = torch.ones((1, past_len + 1), dtype=torch.long, device=input_device)

        with torch.no_grad():
            outputs = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                past_key_values=parent_cache,
                use_cache=True
            )

        logits = outputs.logits[0, -1, :].clone()
        past_key_values = outputs.past_key_values if hasattr(outputs, "past_key_values") else None
        del outputs
        return logits, past_key_values

    async def _get_logits_for_node(self, node: TokenNode) -> torch.Tensor:
        """
        Get next-token logits for a node.

        KV mode OFF:
          - Existing full-prefix batched path.
        KV mode ON:
          - Parent cache hit: one-token forward from parent KV.
          - Cache miss/root: one full-prefix forward and seed KV.
        """
        if not self._kv_cache_enabled:
            prefix = node.get_token_prefix(self.tokenizer, self.model_type)
            tensor = torch.tensor(prefix)
            return await self.lazy_batched_forward(tensor)

        if node.parent is None:
            prefix = node.get_token_prefix(self.tokenizer, self.model_type)
            logits, node_cache = self._forward_full_prefix_with_cache(prefix)
        else:
            parent_cache = self._current_parent_cache.get(node.parent.id)
            if parent_cache is None:
                prefix = node.get_token_prefix(self.tokenizer, self.model_type)
                logits, node_cache = self._forward_full_prefix_with_cache(prefix)
            else:
                logits, node_cache = self._forward_from_parent_cache(node.token_id, parent_cache)

        if node_cache is not None:
            self._store_next_kv_cache(node.id, node_cache)
        return logits

    def _decode_token(self, token_id: int) -> str:
        """Decode a single token."""
        if self.model_type == "llama2":
            raw_token = self.tokenizer.convert_ids_to_tokens([token_id])[0]
            return raw_token.replace("▁", " ")
        return self.tokenizer.decode([token_id])

    def _apply_generation_filters(
        self,
        logits: torch.Tensor,
        temperature: float = 0.7,
        top_p: float = 0.9,
        min_p: float = 0.05
    ) -> torch.Tensor:
        """
        Apply temperature, min_p, and top_p filtering.

        Args:
            logits: Raw logits
            temperature: Temperature scaling
            top_p: Nucleus sampling threshold
            min_p: Minimum probability threshold

        Returns:
            Filtered probability distribution
        """
        if temperature != 1.0:
            logits = logits / temperature

        probs = torch.nn.functional.softmax(logits, dim=-1)

        # Min-p filtering
        if min_p > 0:
            probs = probs.masked_fill(probs < min_p, 0.0)
            if probs.sum() > 0:
                probs = probs / probs.sum()

        # Top-p (nucleus) filtering
        if top_p < 1.0:
            sorted_probs, sorted_indices = torch.sort(probs, descending=True)
            cumsum_probs = torch.cumsum(sorted_probs, dim=-1)
            sorted_indices_to_remove = cumsum_probs > top_p

            sorted_indices_to_remove[1:] = sorted_indices_to_remove[:-1].clone()
            sorted_indices_to_remove[0] = False

            indices_to_remove = sorted_indices_to_remove.scatter(
                0, sorted_indices, sorted_indices_to_remove
            )
            probs = probs.masked_fill(indices_to_remove, 0.0)
            if probs.sum() > 0:
                probs = probs / probs.sum()

        return probs

    async def explore_node(
        self,
        node: TokenNode,
        depth_to_explore: int,
        k: int = 5,
        temperature: float = 0.7,
        top_p: float = 0.9,
        min_p: float = 0.05
    ) -> None:
        """
        Recursively explore a node to specified depth.

        Args:
            node: Node to explore from
            depth_to_explore: Depth to explore
            k: Number of top tokens per step
            temperature: Sampling temperature
            top_p: Nucleus sampling threshold
            min_p: Minimum probability threshold
        """
        async def _explore_recursive(node: TokenNode, depth: int):
            if depth == 0:
                return
            if self._extra_terminator_token_ids:
                if node.token_id in self._extra_terminator_token_ids:
                    return
            elif node.token_id in (self.tokenizer.eos_token_id, 128001, 128008):
                return

            if not node.children:
                logits = await self._get_logits_for_node(node)
                probs = self._apply_generation_filters(logits, temperature, top_p, min_p)

                top_k_actual = min(k, (probs > 0).sum().item())
                if top_k_actual > 0:
                    topk_probs, topk_indices = torch.topk(probs, top_k_actual)

                    for prob, idx in zip(topk_probs, topk_indices):
                        token_id = int(idx.item())
                        token_text = self._decode_token(token_id)
                        prob_value = float(prob.item())

                        child_node = TokenNode(
                            id=os.urandom(4).hex(),
                            token_id=token_id,
                            text=token_text,
                            prob=prob_value,
                            score=prob_value,
                            depth=node.depth + 1,
                            parent=node,
                            children={}
                        )
                        node.children[token_id] = child_node

            if node.children:
                await asyncio.gather(*(
                    _explore_recursive(child, depth - 1)
                    for child in node.children.values()
                ))

        await _explore_recursive(node, depth_to_explore)

        if self._broadcast_queue:
            await self._broadcast_queue.put({
                "type": "update",
                "tree": node.to_dict()
            })
