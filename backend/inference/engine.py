"""Token-level inference engine with batching and advanced generation methods."""

import asyncio
import logging
import math
import os
from typing import Any, List, Optional

import torch
from torch.nn.utils.rnn import pad_sequence
from transformers import AutoModelForCausalLM, AutoTokenizer

from utils.token_node import TokenNode

logger = logging.getLogger(__name__)


class TokenInferenceEngine:
    """
    Token-level inference engine supporting batched forward passes,
    tree exploration, and Sequential Monte Carlo generation.
    """
    
    def __init__(
        self,
        model: AutoModelForCausalLM,
        tokenizer: AutoTokenizer,
        model_type: str = "llama",
        device: str = "cuda:0",
        batch_size: int = 16,
        batch_timeout: float = 0.1
    ):
        """
        Initialize the inference engine.
        
        Args:
            model: The loaded language model.
            tokenizer: The tokenizer for the model.
            model_type: Type of model (llama, qwen, exaone).
            device: Device to run inference on.
            batch_size: Maximum batch size for inference.
            batch_timeout: Timeout for batching requests.
        """
        self.model = model
        self.tokenizer = tokenizer
        self.model_type = model_type
        self.device = device
        self.batch_size = batch_size
        self.batch_timeout = batch_timeout
        
        self.model.to(self.device)
        
        # Batch processing
        self._batch_queue = asyncio.Queue()
        self._broadcast_queue: Optional[asyncio.Queue] = None
        
        # Start background batch worker
        asyncio.create_task(self._run_batch_worker())
        
        logger.info(f"TokenInferenceEngine initialized - Device: {device}, Batch size: {batch_size}")
    
    def set_broadcast_queue(self, queue: asyncio.Queue) -> None:
        """Set the broadcast queue for real-time updates."""
        self._broadcast_queue = queue
    
    async def lazy_batched_forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """
        Register a forward pass request to be batched with others.
        
        Args:
            inputs: Input tensor of token IDs.
            
        Returns:
            torch.Tensor: Logits for the last token position.
        """
        loop = asyncio.get_running_loop()
        future = loop.create_future()
        
        await self._batch_queue.put((inputs, future))
        return await future
    
    async def _run_batch_worker(self) -> None:
        """
        Background worker that batches forward pass requests for efficiency.
        
        Collects requests until batch_size is reached or batch_timeout expires,
        then processes them in a single forward pass.
        """
        while True:
            batch = []
            loop = asyncio.get_running_loop()
            start_time = loop.time()
            
            try:
                # Wait for first request
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
                except Exception as e:
                    logger.error(f"Error in batch collection: {e}")
                    break
            
            # Process batch
            await self._process_inference_batch(batch)
    
    async def _process_inference_batch(self, batch: List[tuple]) -> None:
        """
        Process a batch of forward pass requests.
        
        Args:
            batch: List of (inputs, future) tuples.
        """
        inputs_list = [req[0] for req in batch]
        futures = [req[1] for req in batch]
        
        try:
            # Prepare tensors for batching
            input_ids_list = []
            orig_lengths = []
            
            for inp in inputs_list:
                if inp.ndim == 1:
                    inp = inp.unsqueeze(0)
                orig_lengths.append(inp.shape[-1])
                input_ids_list.append(inp.squeeze(0))
            
            # Pad sequences to same length
            padded_input_ids = pad_sequence(
                input_ids_list, 
                batch_first=True, 
                padding_value=self.tokenizer.eos_token_id
            )
            
            # Forward pass
            with torch.no_grad():
                outputs = self.model(input_ids=padded_input_ids.to(self.device))
            
            # Distribute results
            if hasattr(outputs, "logits"):
                logits = outputs.logits  # [batch_size, seq_len, vocab_size]
                
                for i, (future, orig_len) in enumerate(zip(futures, orig_lengths)):
                    if not future.done():
                        last_token_logits = logits[i, orig_len - 1, :]
                        future.set_result(last_token_logits)
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
    
    def _sample_greedy(self, probs: torch.Tensor) -> int:
        """Sample greedily from probability distribution."""
        probs = probs.cpu()
        return int(torch.argmax(probs).item())
    
    def _sample_from_probs(self, probs: torch.Tensor) -> int:
        """Sample from probability distribution."""
        probs = probs.cpu()
        return int(torch.multinomial(probs, num_samples=1).item())
    
    def _apply_generation_filters(
        self, 
        logits: torch.Tensor,
        temperature: float = 0.7,
        top_p: float = 0.9, 
        min_p: float = 0.05
    ) -> torch.Tensor:
        """
        Apply temperature, min_p, and top_p filtering to logits.
        
        Args:
            logits: Raw logits from utils.
            temperature: Temperature scaling factor.
            top_p: Nucleus sampling threshold.
            min_p: Minimum probability threshold.
            
        Returns:
            torch.Tensor: Filtered probability distribution.
        """
        # Temperature scaling
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
            
            # Keep at least one token
            sorted_indices_to_remove[1:] = sorted_indices_to_remove[:-1].clone()
            sorted_indices_to_remove[0] = False
            
            indices_to_remove = sorted_indices_to_remove.scatter(0, sorted_indices, sorted_indices_to_remove)
            probs = probs.masked_fill(indices_to_remove, 0.0)
            if probs.sum() > 0:
                probs = probs / probs.sum()
        
        return probs
    
    async def generate_with_topk(
        self,
        node: TokenNode,
        k: int = 5,
        max_tokens: int = 50,
        temperature: float = 0.7,
        top_p: float = 0.9,
        min_p: float = 0.05
    ) -> TokenNode:
        """
        Generate tokens using top-k sampling with tree structure.
        
        Args:
            node: Starting node for generation.
            k: Number of top tokens to consider.
            max_tokens: Maximum tokens to generate.
            temperature: Sampling temperature.
            top_p: Nucleus sampling threshold.
            min_p: Minimum probability threshold.
            
        Returns:
            TokenNode: Root node of generated tree.
        """
        inputs = torch.tensor(
            node.get_token_prefix(self.tokenizer, self.model_type), 
            device=self.device
        ).unsqueeze(0)
        current_node = node
        
        for _ in range(max_tokens):
            logits = await self.lazy_batched_forward(inputs)
            probs = self._apply_generation_filters(logits, temperature, top_p, min_p)
            
            # Select top-k tokens
            top_k_actual = min(k, (probs > 0).sum().item())
            if top_k_actual == 0:
                break
                
            topk_probs, topk_indices = torch.topk(probs, top_k_actual)
            
            # Create child nodes
            children = {}
            for prob, idx in zip(topk_probs, topk_indices):
                token_id = int(idx.item())
                token_text = self.tokenizer.decode([token_id])
                prob_value = float(prob.item())
                
                if token_text not in current_node.children:
                    child_node = TokenNode(
                        id=os.urandom(4).hex(),
                        token_id=token_id,
                        text=token_text,
                        prob=prob_value,
                        score=prob_value,
                        depth=current_node.depth + 1,
                        parent=current_node,
                        children={}
                    )
                    children[token_text] = child_node
                else:
                    children[token_text] = current_node.children[token_text]
            
            current_node.children = children
            
            # Select next token greedily
            if probs.sum() > 0:
                next_token_idx = self._sample_greedy(probs)
                next_token_text = self.tokenizer.decode([next_token_idx])
                
                current_node = children.get(next_token_text, children[list(children.keys())[0]])
                new_token = torch.tensor([[next_token_idx]], device=self.device)
                inputs = torch.cat([inputs, new_token], dim=1)
                
                if next_token_idx == self.tokenizer.eos_token_id:
                    break
            else:
                break
        
        return node
    
    async def explore_node(
        self,
        node: TokenNode,
        depth_to_explore: int,
        k: int = 5,
        temperature: float = 0.7,
        top_p: float = 0.9,
        min_p: float = 0.05,
        extend_greedy: bool = True
    ) -> None:
        """
        Recursively explore a node to a specified depth.
        
        Args:
            node: Node to explore from.
            depth_to_explore: Depth to explore.
            k: Number of top tokens per step.
            temperature: Sampling temperature.
            top_p: Nucleus sampling threshold.
            min_p: Minimum probability threshold.
            extend_greedy: Whether to extend greedy path.
        """
        async def _explore_recursive(node: TokenNode, depth: int, is_greedy_path: bool = False):
            if depth == 0 or node.token_id == self.tokenizer.eos_token_id:
                return
            
            if not node.children:
                prefix = node.get_token_prefix(self.tokenizer, self.model_type)
                tensor = torch.tensor(prefix, device=self.device)
                
                logits = await self.lazy_batched_forward(tensor)
                probs = self._apply_generation_filters(logits, temperature, top_p, min_p)
                
                # Create top-k children
                top_k_actual = min(k, (probs > 0).sum().item())
                if top_k_actual > 0:
                    topk_probs, topk_indices = torch.topk(probs, top_k_actual)
                    
                    for prob, idx in zip(topk_probs, topk_indices):
                        token_id = int(idx.item())
                        token_text = self.tokenizer.decode([token_id])
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
                        node.children[token_text] = child_node
            
            if depth == 1 and extend_greedy and is_greedy_path and node.children:
                # Extend greedy path
                greedy_child = max(node.children.values(), key=lambda x: x.prob)
                await self.extend_greedy_path(greedy_child, 12, k, temperature, top_p, min_p)
                return
            
            # Recursive exploration
            if node.children:
                greedy_child = max(node.children.values(), key=lambda x: x.prob)
                await asyncio.gather(*(
                    _explore_recursive(child, depth - 1, is_greedy_path and child == greedy_child)
                    for child in node.children.values()
                ))
        
        await _explore_recursive(node, depth_to_explore, True)
        
        # Broadcast update if queue is available
        if self._broadcast_queue:
            await self._broadcast_queue.put({
                "type": "update",
                "tree": node.to_dict()
            })
    
    async def extend_greedy_path(
        self,
        node: TokenNode,
        greedy_depth: int,
        k: int = 5,
        temperature: float = 0.7,
        top_p: float = 0.9,
        min_p: float = 0.05
    ) -> None:
        """
        Extend a greedy path from the given node.
        
        Args:
            node: Starting node.
            greedy_depth: Depth of greedy path.
            k: Number of alternatives to store at each step.
            temperature: Sampling temperature.
            top_p: Nucleus sampling threshold.
            min_p: Minimum probability threshold.
        """
        current = node
        
        for _ in range(greedy_depth):
            if current.token_id == self.tokenizer.eos_token_id:
                break
            
            if current.children:
                # Continue with existing best child
                current = max(current.children.values(), key=lambda x: x.prob)
                continue
            
            prefix = current.get_token_prefix(self.tokenizer, self.model_type)
            tensor = torch.tensor(prefix, device=self.device)
            
            logits = await self.lazy_batched_forward(tensor)
            probs = self._apply_generation_filters(logits, temperature, top_p, min_p)
            
            # Create top-k children
            top_k_actual = min(k, (probs > 0).sum().item())
            if top_k_actual == 0:
                break
            
            topk_probs, topk_indices = torch.topk(probs, top_k_actual)
            
            best_child = None
            for i, (prob, idx) in enumerate(zip(topk_probs, topk_indices)):
                token_id = int(idx.item())
                token_text = self.tokenizer.decode([token_id])
                prob_value = float(prob.item())
                
                child_node = TokenNode(
                    id=os.urandom(4).hex(),
                    token_id=token_id,
                    text=token_text,
                    prob=prob_value,
                    score=prob_value,
                    depth=current.depth + 1,
                    parent=current,
                    children={}
                )
                
                current.children[token_text] = child_node
                
                if i == 0:  # Best candidate
                    best_child = child_node
            
            current = best_child
    
    async def generate_with_smc(
        self,
        node: TokenNode,
        k: int = 5,
        particlenum: int = 20,
        max_tokens: int = 15,
        temperature: float = 0.7,
        top_p: float = 0.9,
        min_p: float = 0.05
    ) -> TokenNode:
        """
        Generate text using Sequential Monte Carlo (SMC) sampling.
        
        Args:
            node: Starting node.
            k: Number of candidates per step.
            particlenum: Number of particles.
            max_tokens: Maximum tokens to generate.
            temperature: Sampling temperature.
            top_p: Nucleus sampling threshold.
            min_p: Minimum probability threshold.
            
        Returns:
            TokenNode: Root node with generated tree.
        """
        # Initialize particles
        initial_seq = torch.tensor(
            node.get_token_prefix(self.tokenizer, self.model_type),
            device=self.device
        ).unsqueeze(0)
        
        particles = []
        for _ in range(particlenum):
            particles.append({
                "seq": initial_seq.clone(),
                "node": node,
                "log_weight": 0.0,
                "terminated": False
            })
        
        for t in range(max_tokens):
            active_particles = [p for p in particles if not p["terminated"]]
            if not active_particles:
                break
            
            # Batch forward passes for active particles
            forward_tasks = [self.lazy_batched_forward(p["seq"]) for p in active_particles]
            logits_list = await asyncio.gather(*forward_tasks)
            
            # Process each particle
            for idx, p in enumerate(active_particles):
                logits = logits_list[idx]
                probs = self._apply_generation_filters(logits, temperature, top_p, min_p)
                
                nonzero_indices = (probs > 0).nonzero(as_tuple=False).view(-1)
                if nonzero_indices.numel() == 0:
                    p["terminated"] = True
                    continue
                
                candidate_count = min(k, nonzero_indices.numel())
                topk_probs, topk_indices = torch.topk(probs, candidate_count)
                candidate_distribution = topk_probs / (topk_probs.sum() + 1e-12)
                
                # Create candidate nodes
                candidate_nodes = {}
                for prob_val, idx_val in zip(topk_probs, topk_indices):
                    token_idx = int(idx_val.item())
                    token_prob = float(prob_val.item())
                    token_text = self.tokenizer.decode([token_idx])
                    
                    if token_text in p["node"].children:
                        candidate_node = p["node"].children[token_text]
                    else:
                        candidate_node = TokenNode(
                            id=os.urandom(4).hex(),
                            token_id=token_idx,
                            text=token_text,
                            prob=token_prob,
                            score=token_prob,
                            depth=p["node"].depth + 1,
                            parent=p["node"],
                            children={}
                        )
                        p["node"].children[token_text] = candidate_node
                    
                    candidate_nodes[token_idx] = candidate_node
                
                # Sample from candidates
                sampled_in_topk = torch.multinomial(candidate_distribution, num_samples=1)
                sampled_token_idx = int(topk_indices[sampled_in_topk].item())
                token_prob = float(probs[sampled_token_idx].item())
                
                # Update particle
                p["log_weight"] += math.log(token_prob + 1e-12)
                new_token = torch.tensor([[sampled_token_idx]], device=self.device)
                p["seq"] = torch.cat([p["seq"], new_token], dim=1)
                p["node"] = candidate_nodes[sampled_token_idx]
                
                if sampled_token_idx == self.tokenizer.eos_token_id:
                    p["terminated"] = True
            
            # Resampling based on effective sample size
            active_particles = [p for p in particles if not p["terminated"]]
            if active_particles:
                weights = [math.exp(p["log_weight"]) for p in active_particles]
                total_weight = sum(weights)
                normalized_weights = [w / (total_weight + 1e-12) for w in weights]
                ess = 1.0 / sum(w ** 2 for w in normalized_weights)
                
                if ess < len(active_particles) / 2.0:
                    # Resample particles
                    indices = torch.multinomial(
                        torch.tensor(normalized_weights), 
                        num_samples=len(active_particles), 
                        replacement=True
                    )
                    
                    new_particles = []
                    active_idx = 0
                    for p in particles:
                        if p["terminated"]:
                            new_particles.append(p)
                        else:
                            sampled_particle = active_particles[indices[active_idx].item()]
                            new_particle = {
                                "seq": sampled_particle["seq"].clone(),
                                "node": sampled_particle["node"],
                                "log_weight": 0.0,
                                "terminated": sampled_particle["terminated"]
                            }
                            new_particles.append(new_particle)
                            active_idx += 1
                    particles = new_particles
        
        return node