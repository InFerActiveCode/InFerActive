"""In-process vLLM inference engine for interactive BFS tree generation."""

import asyncio
import gc
import logging
import math
import os
from pathlib import Path
from typing import Any, Iterable, List, Optional, Sequence, Tuple

from utils.token_node import TokenNode

try:
    import torch
except ModuleNotFoundError:  # pragma: no cover - optional on mock/dev machines
    torch = None

try:
    from transformers import AutoTokenizer
except ModuleNotFoundError:  # pragma: no cover - optional on mock/dev machines
    AutoTokenizer = None

try:
    from vllm import LLM, SamplingParams
except ModuleNotFoundError:  # pragma: no cover - optional on mock/dev machines
    LLM = None
    SamplingParams = None


logger = logging.getLogger(__name__)


LLAMA2_SYSTEM_PROMPT = """You are a helpful, respectful and honest assistant. Always answer as helpfully as possible, while being safe. Your answers should not include any harmful, unethical, racist, sexist, toxic, dangerous, or illegal content. Please ensure that your responses are socially unbiased and positive in nature.

If a question does not make any sense, or is not factually coherent, explain why instead of answering something not correct. If you don't know the answer to a question, please don't share false information."""


class VLLMInferenceEngine:
    """vLLM-backed engine used by the WebSocket backend.

    The engine mirrors the evaluation pipeline's one-token processed-logprob
    expansion, but keeps all tree state in memory for interactive frontend
    updates.
    """

    is_vllm = True

    def __init__(
        self,
        model_path: Path,
        model_type: str = "llama3",
        dtype: str = "float16",
        tensor_parallel_size: int = 1,
        max_model_len: int = 2048,
        gpu_memory_utilization: float = 0.7,
        batch_size: int = 128,
        max_logprobs: int = 200,
        enable_prefix_caching: bool = True,
        max_depth: int = 512,
        max_leaves: int = 1000,
    ):
        self.model_path = str(model_path)
        self.model_type = model_type.lower()
        self.dtype = dtype
        self.tensor_parallel_size = tensor_parallel_size
        self.max_model_len = max_model_len
        self.gpu_memory_utilization = gpu_memory_utilization
        self.batch_size = batch_size
        self.max_logprobs = max_logprobs
        self.enable_prefix_caching = enable_prefix_caching
        self.max_depth = max_depth
        self.max_leaves = max_leaves

        self.llm: Optional[Any] = None
        self.tokenizer: Optional[Any] = None
        self.terminator_token_ids: List[int] = []
        self._broadcast_queue: Optional[asyncio.Queue] = None
        self._generate_lock = asyncio.Lock()

    async def initialize(self) -> None:
        """Load tokenizer and in-process vLLM engine."""
        if AutoTokenizer is None or LLM is None or SamplingParams is None:
            raise RuntimeError(
                "vLLM backend requires transformers and vllm. "
                "Install the CUDA vLLM environment before using --backend vllm."
            )

        logger.info("Loading tokenizer from %s", self.model_path)
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_path,
            use_fast=True,
            trust_remote_code=True,
        )
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self.terminator_token_ids = self._get_terminator_token_ids()

        logger.info("Loading in-process vLLM engine from %s", self.model_path)
        self.llm = LLM(
            model=self.model_path,
            dtype=self.dtype,
            tensor_parallel_size=self.tensor_parallel_size,
            max_model_len=self.max_model_len,
            gpu_memory_utilization=self.gpu_memory_utilization,
            trust_remote_code=True,
            enable_prefix_caching=self.enable_prefix_caching,
            logprobs_mode="processed_logprobs",
            max_logprobs=self.max_logprobs,
        )
        logger.info("vLLM engine initialized")

    def cleanup(self) -> None:
        """Release vLLM resources."""
        if self.llm is not None:
            del self.llm
            self.llm = None
        if self.tokenizer is not None:
            del self.tokenizer
            self.tokenizer = None
        gc.collect()
        if torch is not None and torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()

    def set_broadcast_queue(self, queue: asyncio.Queue) -> None:
        """Set the broadcast queue for subtree updates."""
        self._broadcast_queue = queue

    async def generate_with_bfs(
        self,
        node: TokenNode,
        k: int = 5,
        particlenum: int = 20,
        max_tokens: int = 15,
        temperature: float = 0.7,
        top_p: float = 0.9,
        min_p: float = 0.0,
        depth: Optional[int] = None,
    ) -> TokenNode:
        """Initialize a root tree with breadth-first one-token expansions."""
        del particlenum
        fallback_depth = min(max_tokens, 3)
        depth_limit = self._resolve_depth(depth, fallback=fallback_depth)
        depth_limit = min(depth_limit, self.max_depth)

        node.children = {}
        frontier = [node]
        await self._expand_bfs(
            root=node,
            frontier=frontier,
            depth_limit=depth_limit,
            k=k,
            temperature=temperature,
            top_p=top_p,
            min_p=min_p,
        )
        return node

    async def generate_with_smc(self, *args: Any, **kwargs: Any) -> TokenNode:
        """Compatibility alias for older clients."""
        return await self.generate_with_bfs(*args, **kwargs)

    async def explore_node(
        self,
        node: TokenNode,
        depth_to_explore: int,
        k: int = 5,
        temperature: float = 0.7,
        top_p: float = 0.9,
        min_p: float = 0.0,
        extend_greedy: bool = True,
    ) -> None:
        """Expand the leaf frontier under a selected node and broadcast it."""
        del extend_greedy
        depth_limit = min(self._resolve_depth(depth_to_explore, fallback=1), self.max_depth)
        frontier = self._expandable_leaves(node)
        if not frontier and self._is_expandable(node):
            frontier = [node]

        await self._expand_bfs(
            root=node,
            frontier=frontier,
            depth_limit=depth_limit,
            k=k,
            temperature=temperature,
            top_p=top_p,
            min_p=min_p,
        )

        if self._broadcast_queue:
            await self._broadcast_queue.put({
                "type": "update",
                "tree": node.to_dict(),
            })

    async def _expand_bfs(
        self,
        root: TokenNode,
        frontier: Sequence[TokenNode],
        depth_limit: int,
        k: int,
        temperature: float,
        top_p: float,
        min_p: float,
    ) -> None:
        current_frontier = list(frontier)
        for _ in range(depth_limit):
            active_nodes = [node for node in current_frontier if self._is_expandable(node)]
            if not active_nodes:
                break

            next_frontier: List[TokenNode] = []
            for start in range(0, len(active_nodes), self.batch_size):
                batch = active_nodes[start:start + self.batch_size]
                await self._expand_batch(batch, k, temperature, top_p, min_p)
                for expanded_node in batch:
                    next_frontier.extend(expanded_node.children.values())

            leaves_before_prune = len(root.collect_leaf_nodes())
            self._prune_to_max_leaves(root)
            if self.max_leaves > 0 and leaves_before_prune > self.max_leaves:
                break
            current_frontier = next_frontier

    async def _expand_batch(
        self,
        nodes: Sequence[TokenNode],
        k: int,
        temperature: float,
        top_p: float,
        min_p: float,
    ) -> None:
        async with self._generate_lock:
            loop = asyncio.get_running_loop()
            await loop.run_in_executor(
                None,
                self._expand_batch_sync,
                list(nodes),
                k,
                temperature,
                top_p,
                min_p,
            )

    def _expand_batch_sync(
        self,
        nodes: Sequence[TokenNode],
        k: int,
        temperature: float,
        top_p: float,
        min_p: float,
    ) -> None:
        if self.llm is None or self.tokenizer is None:
            raise RuntimeError("vLLM engine is not initialized")
        if self.max_logprobs != -1 and k > self.max_logprobs:
            raise ValueError(
                f"k={k} exceeds max_logprobs={self.max_logprobs}. "
                "Increase --max-logprobs or lower --k."
            )

        prompts = [
            {"prompt_token_ids": self._get_token_prefix(node)}
            for node in nodes
        ]
        outputs = self.llm.generate(
            prompts,
            sampling_params=self._build_sampling_params(k, temperature, top_p, min_p),
        )

        for node, output in zip(nodes, outputs):
            if not output.outputs:
                continue
            candidates = self._extract_candidates(output.outputs[0].logprobs)
            for token_id, logprob, _rank, decoded_token in candidates:
                token_text = decoded_token if decoded_token is not None else self._decode_token(token_id)
                prob_value = float(math.exp(logprob))
                child_key = str(token_id)
                child_node = node.children.get(child_key)
                if child_node is None:
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
                    node.children[child_key] = child_node
                else:
                    child_node.text = token_text
                    child_node.prob = prob_value
                    child_node.score = prob_value

    def _build_sampling_params(
        self,
        k: int,
        temperature: float,
        top_p: float,
        min_p: float,
    ) -> Any:
        if SamplingParams is None:
            raise RuntimeError("vLLM is not installed")
        if self.max_logprobs == -1:
            request_logprobs = -1
        elif k > 0:
            request_logprobs = min(k, self.max_logprobs)
        else:
            request_logprobs = self.max_logprobs
        return SamplingParams(
            n=1,
            max_tokens=1,
            temperature=temperature,
            top_p=top_p,
            top_k=k if k > 0 else 0,
            min_p=min_p,
            logprobs=request_logprobs,
            detokenize=True,
            skip_special_tokens=False,
            ignore_eos=True,
        )

    def _extract_candidates(self, sample_logprobs: Any) -> List[Tuple[int, float, Optional[int], Optional[str]]]:
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

        ranked = [candidate for candidate in candidates if candidate[2] is not None]
        if ranked:
            ranked.sort(key=lambda item: (item[2], -item[1], item[0]))
            return ranked

        candidates.sort(key=lambda item: (-item[1], item[0]))
        return candidates

    def _iter_logprob_items(self, logprobs_one_position: Any) -> Iterable[Tuple[int, Any]]:
        if logprobs_one_position is None:
            return []
        if hasattr(logprobs_one_position, "items"):
            return list(logprobs_one_position.items())
        return []

    def _get_token_prefix(self, node: TokenNode) -> List[int]:
        if self.tokenizer is None:
            raise RuntimeError("Tokenizer is not initialized")
        path = node.trace_path_to_root()
        if not path:
            return []
        root = path[0]
        prompt_ids = self._build_root_prompt_ids(root.text)
        return prompt_ids + [path_node.token_id for path_node in path[1:]]

    def _build_root_prompt_ids(self, text: str) -> List[int]:
        assert self.tokenizer is not None
        prompt = self._build_prompt(text)
        return self.tokenizer.encode(prompt, add_special_tokens=True)

    def _build_prompt(self, text: str) -> str:
        assert self.tokenizer is not None

        if self.model_type in {"llama3", "gemma"}:
            try:
                prompt = self.tokenizer.apply_chat_template(
                    [
                        {"role": "system", "content": "You are a helpful assistant."},
                        {"role": "user", "content": text},
                    ],
                    tokenize=False,
                    add_generation_prompt=True,
                )
                return self._strip_leading_bos(prompt)
            except Exception:
                pass

        if self.model_type == "qwen":
            try:
                return self.tokenizer.apply_chat_template(
                    [{"role": "user", "content": text}],
                    tokenize=False,
                    add_generation_prompt=True,
                )
            except Exception:
                pass

        if self.model_type == "llama2":
            return f"[INST] <<SYS>>\n{LLAMA2_SYSTEM_PROMPT}\n<</SYS>>\n\n{text} [/INST] "

        if self.model_type == "llama3":
            return (
                "<|start_header_id|>system<|end_header_id|>\n\n"
                "You are a helpful assistant<|eot_id|>"
                "<|start_header_id|>user<|end_header_id|>\n\n"
                f"{text}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"
            )

        return text

    def _strip_leading_bos(self, prompt: str) -> str:
        assert self.tokenizer is not None
        bos_token = getattr(self.tokenizer, "bos_token", None)
        if bos_token and prompt.startswith(bos_token):
            return prompt.replace(bos_token, "", 1)
        return prompt

    def _decode_token(self, token_id: int) -> str:
        assert self.tokenizer is not None
        if self.model_type == "llama2":
            raw_token = self.tokenizer.convert_ids_to_tokens([token_id])[0]
            return raw_token.replace("▁", " ")
        return self.tokenizer.decode([token_id], skip_special_tokens=False)

    def _get_terminator_token_ids(self) -> List[int]:
        assert self.tokenizer is not None
        terminator_ids: List[int] = []
        eos_token_id = getattr(self.tokenizer, "eos_token_id", None)

        if isinstance(eos_token_id, int):
            terminator_ids.append(eos_token_id)
        elif isinstance(eos_token_id, (list, tuple)):
            terminator_ids.extend(int(token_id) for token_id in eos_token_id)

        if self.model_type == "llama3":
            terminator_ids.extend([128001, 128008, 128009])
        elif self.model_type == "qwen":
            endoftext_token_id = self.tokenizer.convert_tokens_to_ids("<|endoftext|>")
            if isinstance(endoftext_token_id, int) and endoftext_token_id >= 0:
                terminator_ids.append(endoftext_token_id)
        elif self.model_type == "gemma":
            end_of_turn_id = self.tokenizer.convert_tokens_to_ids("<end_of_turn>")
            if isinstance(end_of_turn_id, int) and end_of_turn_id >= 0:
                terminator_ids.append(end_of_turn_id)

        return list(dict.fromkeys(terminator_ids))

    def _is_expandable(self, node: TokenNode) -> bool:
        return (
            node.token_id not in self.terminator_token_ids
            and node.depth < self.max_depth
        )

    def _expandable_leaves(self, node: TokenNode) -> List[TokenNode]:
        return [leaf for leaf in node.collect_leaf_nodes() if self._is_expandable(leaf)]

    def _prune_to_max_leaves(self, root: TokenNode) -> int:
        if self.max_leaves <= 0:
            return len(root.collect_leaf_nodes())

        leaves = root.collect_leaf_nodes()
        if len(leaves) <= self.max_leaves:
            return len(leaves)

        max_leaf_depth = max(node.depth for node in leaves)
        targets = [node for node in leaves if node.depth == max_leaf_depth]
        targets.reverse()

        to_remove = len(leaves) - self.max_leaves
        for node in targets:
            if to_remove <= 0:
                break
            parent = node.parent
            if parent is None:
                continue

            remove_key = None
            for key, child in parent.children.items():
                if child is node:
                    remove_key = key
                    break
            if remove_key is None:
                continue

            is_only_child = len(parent.children) == 1
            del parent.children[remove_key]
            if not is_only_child:
                to_remove -= 1

        return len(root.collect_leaf_nodes())

    @staticmethod
    def _resolve_depth(value: Optional[int], fallback: int) -> int:
        try:
            if value is None:
                return max(0, int(fallback))
            return max(0, int(value))
        except (TypeError, ValueError):
            return max(0, int(fallback))
