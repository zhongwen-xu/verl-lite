# Copyright 2025 Tencent
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Local FSDP workers for verl-lite using actual PyTorch FSDP implementation.

These workers provide the same APIs as verl's Ray-based workers but use
PyTorch FSDP directly without Ray dependencies.
"""

import logging
import os
from contextlib import nullcontext
from typing import Any, Dict, Generator, Optional, Tuple

import torch
import torch.nn as nn
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.optim import AdamW
from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig
from tensordict import TensorDict

# Import verl components we can reuse
from verl import DataProto

# Try to import verl utilities with fallbacks
try:
    from verl.utils.fsdp_utils import (
        MixedPrecisionPolicy, 
        CPUOffloadPolicy,
        get_fsdp_wrap_policy,
        init_fn,
        apply_fsdp2
    )
except ImportError:
    MixedPrecisionPolicy = None
    CPUOffloadPolicy = None
    def get_fsdp_wrap_policy():
        return None
    def init_fn(module):
        return module
    def apply_fsdp2(model, config):
        return model

try:
    from verl.utils.model import compute_position_id_with_mask
except ImportError:
    def compute_position_id_with_mask(input_ids, attention_mask):
        return torch.arange(input_ids.size(1), device=input_ids.device).expand_as(input_ids)

try:
    from verl.utils.torch_functional import masked_mean, get_response_mask
except ImportError:
    def masked_mean(tensor, mask, dim=None):
        masked_tensor = tensor * mask
        return masked_tensor.sum(dim=dim) / mask.sum(dim=dim).clamp(min=1)
    
    def get_response_mask(response_ids, pad_token_id):
        return (response_ids != pad_token_id).float()

try:
    from verl.workers.config import FSDPActorConfig, FSDPCriticConfig, RolloutConfig
except ImportError:
    from verl.workers.fsdp_workers import FSDPActorConfig, FSDPCriticConfig
    class RolloutConfig:
        def __init__(self, **kwargs):
            for k, v in kwargs.items():
                setattr(self, k, v)

try:
    from verl.trainer.ppo.core_algos import compute_log_probs
except ImportError:
    def compute_log_probs(logits, input_ids, attention_mask):
        # Simple log prob computation
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = input_ids[..., 1:].contiguous()
        shift_mask = attention_mask[..., 1:].contiguous()
        
        log_probs = torch.log_softmax(shift_logits, dim=-1)
        selected_log_probs = log_probs.gather(-1, shift_labels.unsqueeze(-1)).squeeze(-1)
        return selected_log_probs * shift_mask

logger = logging.getLogger(__name__)


class LocalFSDPActorWorker:
    """
    Local FSDP Actor worker using actual PyTorch FSDP.
    
    This maintains the same API as verl's ActorRolloutRefWorker but runs
    locally with PyTorch FSDP instead of Ray.
    """
    
    def __init__(self, config: FSDPActorConfig):
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Initialize model and tokenizer
        self._load_model_and_tokenizer()
        
        # Initialize optimizer
        self._init_optimizer()
        
        # Training state
        self.global_step = 0
        
        logger.info(f"Initialized LocalFSDPActorWorker on device: {self.device}")
    
    def _load_model_and_tokenizer(self):
        """Load model and tokenizer, apply FSDP."""
        model_path = self.config.model.path
        
        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # Load model
        with init_fn():
            model = AutoModelForCausalLM.from_pretrained(
                model_path,
                torch_dtype=torch.float16 if self.device.type == "cuda" else torch.float32,
                trust_remote_code=True
            )
        
        # Apply FSDP
        wrap_policy = get_fsdp_wrap_policy(model, self.config.fsdp_config)
        
        # Mixed precision policy
        mp_policy = None
        if hasattr(self.config.fsdp_config, 'mixed_precision') and self.config.fsdp_config.mixed_precision:
            mp_policy = MixedPrecisionPolicy(
                param_dtype=torch.float16 if self.device.type == "cuda" else torch.float32,
                reduce_dtype=torch.float16 if self.device.type == "cuda" else torch.float32,
                buffer_dtype=torch.float16 if self.device.type == "cuda" else torch.float32,
            )
        
        # CPU offload policy
        cpu_offload = None
        if hasattr(self.config.fsdp_config, 'param_offload') and self.config.fsdp_config.param_offload:
            cpu_offload = CPUOffloadPolicy(offload_params=True)
        
        # Apply FSDP
        self.model = FSDP(
            model,
            auto_wrap_policy=wrap_policy,
            mixed_precision=mp_policy,
            cpu_offload=cpu_offload,
            device_id=self.device if self.device.type == "cuda" else None,
            sync_module_states=True,
        )
        
        self.model.train()
        logger.info(f"Applied FSDP to model: {model_path}")
    
    def _init_optimizer(self):
        """Initialize optimizer for FSDP model."""
        optimizer_config = self.config.actor.optim
        
        self.optimizer = AdamW(
            self.model.parameters(),
            lr=optimizer_config.lr,
            betas=(optimizer_config.beta1, optimizer_config.beta2),
            weight_decay=optimizer_config.weight_decay,
            eps=optimizer_config.eps,
        )
        
        logger.info(f"Initialized optimizer with lr={optimizer_config.lr}")
    
    def generate_sequences(self, prompts: DataProto) -> DataProto:
        """Generate sequences using the actor model."""
        self.model.eval()
        
        with torch.no_grad():
            input_ids = prompts.batch["input_ids"].to(self.device)
            attention_mask = prompts.batch["attention_mask"].to(self.device)
            
            # Generation config
            gen_config = GenerationConfig(
                max_new_tokens=getattr(self.config.rollout, 'response_length', 256),
                temperature=getattr(self.config.rollout, 'temperature', 1.0),
                top_p=getattr(self.config.rollout, 'top_p', 1.0),
                do_sample=getattr(self.config.rollout, 'do_sample', True),
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
            )
            
            # Generate
            outputs = self.model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                generation_config=gen_config,
                return_dict_in_generate=True,
                output_scores=True,
            )
            
            sequences = outputs.sequences  # Full sequences
            prompt_length = input_ids.size(1)
            response_ids = sequences[:, prompt_length:]  # Only response part
            
            # Create response mask
            response_mask = get_response_mask(response_ids, self.tokenizer.pad_token_id)
            
            # Create output batch
            output_batch = TensorDict({
                "input_ids": input_ids.cpu(),
                "attention_mask": attention_mask.cpu(),
                "response_ids": response_ids.cpu(),
                "response_mask": response_mask.cpu(),
                "sequences": sequences.cpu(),
            }, batch_size=(len(prompts),))
            
            # Create response DataProto
            response_data = DataProto(
                batch=output_batch,
                non_tensor_batch=prompts.non_tensor_batch.copy(),
                meta_info=prompts.meta_info.copy()
            )
        
        self.model.train()
        logger.info(f"Generated {len(response_data)} sequences")
        return response_data
    
    def compute_log_probs(self, data: DataProto) -> DataProto:
        """Compute log probabilities for sequences."""
        input_ids = data.batch["input_ids"].to(self.device)
        attention_mask = data.batch["attention_mask"].to(self.device)
        
        if "response_ids" in data.batch:
            response_ids = data.batch["response_ids"].to(self.device)
            full_ids = torch.cat([input_ids, response_ids], dim=-1)
            full_attention_mask = torch.cat([
                attention_mask,
                data.batch["response_mask"].to(self.device)
            ], dim=-1)
        else:
            full_ids = input_ids
            full_attention_mask = attention_mask
        
        # Forward pass
        with torch.cuda.amp.autocast() if self.device.type == "cuda" else nullcontext():
            outputs = self.model(
                input_ids=full_ids,
                attention_mask=full_attention_mask,
                return_dict=True,
            )
            logits = outputs.logits
        
        # Compute log probs
        log_probs = compute_log_probs(logits, full_ids, full_attention_mask)
        
        # Add to output
        output_batch = data.batch.clone()
        output_batch["log_probs"] = log_probs.cpu()
        
        return DataProto(
            batch=output_batch,
            non_tensor_batch=data.non_tensor_batch,
            meta_info=data.meta_info
        )
    
    def update_policy(self, data: DataProto) -> Dict[str, Any]:
        """Update policy with PPO mini-batch training."""
        self.model.train()
        
        # Configuration parameters
        ppo_epochs = getattr(self.config.actor, 'ppo_epochs', 4)
        ppo_mini_batch_size = getattr(self.config.actor, 'ppo_mini_batch_size', 32)
        ppo_micro_batch_size = getattr(self.config.actor, 'ppo_micro_batch_size_per_gpu', 4)
        grad_clip = getattr(self.config.actor, 'grad_clip', 1.0)
        
        # Split data into mini-batches
        mini_batches = data.split(ppo_mini_batch_size)
        
        # Metrics accumulation
        all_metrics = []
        
        # Multiple PPO epochs over the same data
        for _ in range(ppo_epochs):
            epoch_metrics = []
            
            for _, mini_batch in enumerate(mini_batches):
                # Further split mini-batch into micro-batches for gradient accumulation
                micro_batches = mini_batch.split(ppo_micro_batch_size)
                gradient_accumulation_steps = len(micro_batches)
                
                # Zero gradients at start of mini-batch
                self.optimizer.zero_grad()
                
                # Process each micro-batch
                micro_metrics = []
                for micro_batch in micro_batches:
                    metrics = self._train_micro_batch(micro_batch, gradient_accumulation_steps)
                    micro_metrics.append(metrics)
                
                # Gradient clipping and optimizer step
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), grad_clip)
                self.optimizer.step()
                self.global_step += 1
                
                # Average metrics across micro-batches
                batch_metrics = {}
                for key in micro_metrics[0].keys():
                    batch_metrics[key] = sum(m[key] for m in micro_metrics) / len(micro_metrics)
                batch_metrics["global_step"] = self.global_step
                
                epoch_metrics.append(batch_metrics)
            
            all_metrics.extend(epoch_metrics)
        
        # Average metrics across all mini-batches and epochs
        final_metrics = {}
        for key in all_metrics[0].keys():
            if key != "global_step":
                final_metrics[key] = sum(m[key] for m in all_metrics) / len(all_metrics)
        final_metrics["global_step"] = self.global_step
        final_metrics["ppo_epochs"] = ppo_epochs
        final_metrics["num_mini_batches"] = len(mini_batches)
        
        logger.info(f"Policy update complete. Step {self.global_step}, "
                   f"epochs: {ppo_epochs}, mini-batches: {len(mini_batches)}, "
                   f"avg loss: {final_metrics.get('total_loss', 0):.4f}")
        
        return final_metrics
    
    def _train_micro_batch(self, micro_batch: DataProto, gradient_accumulation_steps: int) -> Dict[str, Any]:
        """Train on a single micro-batch with gradient accumulation."""
        # Move to device
        input_ids = micro_batch.batch["input_ids"].to(self.device)
        attention_mask = micro_batch.batch["attention_mask"].to(self.device)
        
        if "response_ids" in micro_batch.batch:
            response_ids = micro_batch.batch["response_ids"].to(self.device)
            response_mask = micro_batch.batch["response_mask"].to(self.device)
            full_ids = torch.cat([input_ids, response_ids], dim=-1)
            full_attention_mask = torch.cat([attention_mask, response_mask], dim=-1)
        else:
            full_ids = input_ids
            full_attention_mask = attention_mask
        
        # Get training targets
        advantages = micro_batch.batch.get("advantages", torch.zeros_like(full_ids[:, :-1]))
        old_log_probs = micro_batch.batch.get("old_log_probs", torch.zeros_like(full_ids[:, :-1]))
        
        advantages = advantages.to(self.device)
        old_log_probs = old_log_probs.to(self.device)
        
        # Forward pass
        with torch.cuda.amp.autocast() if self.device.type == "cuda" else nullcontext():
            outputs = self.model(
                input_ids=full_ids,
                attention_mask=full_attention_mask,
                return_dict=True,
            )
            logits = outputs.logits
            
            # Compute current log probs
            new_log_probs = compute_log_probs(logits, full_ids, full_attention_mask)
            
            # PPO loss computation
            ratio = torch.exp(new_log_probs - old_log_probs)
            clip_ratio = torch.clamp(
                ratio,
                1.0 - self.config.actor.ppo_eps,
                1.0 + self.config.actor.ppo_eps
            )
            
            # Policy loss
            policy_loss1 = ratio * advantages
            policy_loss2 = clip_ratio * advantages
            policy_loss = -torch.min(policy_loss1, policy_loss2).mean()
            
            # Entropy loss
            entropy = -(torch.exp(new_log_probs) * new_log_probs).sum(-1).mean()
            entropy_loss = -self.config.actor.entropy_coeff * entropy
            
            # Total loss with gradient accumulation scaling
            total_loss = (policy_loss + entropy_loss) / gradient_accumulation_steps
        
        # Backward pass (accumulate gradients)
        total_loss.backward()
        
        # Return unscaled metrics for proper averaging
        return {
            "policy_loss": policy_loss.item(),
            "entropy_loss": entropy_loss.item(),
            "total_loss": (policy_loss + entropy_loss).item(),  # Unscaled for metrics
            "entropy": entropy.item(),
            "mean_ratio": ratio.mean().item(),
            "kl_divergence": (new_log_probs - old_log_probs).mean().item(),
        }

    def train_step(self, data: DataProto) -> Dict[str, Any]:
        """Legacy method for compatibility. Use update_policy instead."""
        return self.update_policy(data)
    
    def save_checkpoint(self, path: str):
        """Save model and optimizer checkpoints."""
        os.makedirs(path, exist_ok=True)
        
        # Save model
        model_path = os.path.join(path, "model")
        self.model.save_pretrained(model_path)
        self.tokenizer.save_pretrained(model_path)
        
        # Save optimizer
        optim_path = os.path.join(path, "optimizer.pt")
        torch.save({
            "optimizer_state_dict": self.optimizer.state_dict(),
            "global_step": self.global_step,
        }, optim_path)
        
        logger.info(f"Saved checkpoint to {path}")
    
    def load_checkpoint(self, path: str):
        """Load model and optimizer checkpoints."""
        # Load optimizer
        optim_path = os.path.join(path, "optimizer.pt")
        if os.path.exists(optim_path):
            checkpoint = torch.load(optim_path, map_location=self.device)
            self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
            self.global_step = checkpoint["global_step"]
            logger.info(f"Loaded checkpoint from {path}")


class LocalFSDPCriticWorker:
    """
    Local FSDP Critic worker using actual PyTorch FSDP.
    
    Similar to actor but optimized for value function learning.
    """
    
    def __init__(self, config: FSDPCriticConfig):
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Initialize model (reuse actor setup for now)
        self._load_model_and_tokenizer()
        self._init_optimizer()
        
        self.global_step = 0
        logger.info(f"Initialized LocalFSDPCriticWorker on device: {self.device}")
    
    def _load_model_and_tokenizer(self):
        """Load critic model - similar to actor."""
        # Simplified - would need critic-specific model loading
        model_path = self.config.model.path
        
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        with init_fn():
            model = AutoModelForCausalLM.from_pretrained(
                model_path,
                torch_dtype=torch.float16 if self.device.type == "cuda" else torch.float32,
                trust_remote_code=True
            )
        
        # Apply FSDP (similar to actor)
        wrap_policy = get_fsdp_wrap_policy(model, self.config.fsdp_config)
        self.model = FSDP(
            model,
            auto_wrap_policy=wrap_policy,
            device_id=self.device if self.device.type == "cuda" else None,
            sync_module_states=True,
        )
        
        self.model.train()
    
    def _init_optimizer(self):
        """Initialize critic optimizer."""
        optimizer_config = self.config.critic.optim
        
        self.optimizer = AdamW(
            self.model.parameters(),
            lr=optimizer_config.lr,
            betas=(optimizer_config.beta1, optimizer_config.beta2),
            weight_decay=optimizer_config.weight_decay,
            eps=optimizer_config.eps,
        )
    
    def compute_values(self, data: DataProto) -> DataProto:
        """Compute value function estimates."""
        input_ids = data.batch["input_ids"].to(self.device)
        attention_mask = data.batch["attention_mask"].to(self.device)
        
        with torch.no_grad():
            outputs = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                return_dict=True,
            )
            
            # Use last token's hidden state for value (simplified)
            # In practice, you'd want a proper value head
            hidden_states = outputs.hidden_states[-1] if hasattr(outputs, 'hidden_states') else outputs.last_hidden_state
            
            # Simple value computation (last token)
            values = hidden_states[attention_mask.bool()].mean(dim=1, keepdim=True)
            
            output_batch = data.batch.clone()
            output_batch["values"] = values.cpu()
        
        return DataProto(
            batch=output_batch,
            non_tensor_batch=data.non_tensor_batch,
            meta_info=data.meta_info
        )
    
    def train_step(self, data: DataProto) -> Dict[str, Any]:
        """Train the critic."""
        # Simplified critic training
        values = self.compute_values(data)
        target_values = data.batch.get("target_values", torch.zeros_like(values.batch["values"]))
        
        value_loss = nn.MSELoss()(values.batch["values"], target_values.to(self.device))
        
        self.optimizer.zero_grad()
        value_loss.backward()
        self.optimizer.step()
        self.global_step += 1
        
        return {
            "value_loss": value_loss.item(),
            "global_step": self.global_step,
        }


class LocalFSDPWorkers:
    """
    Local orchestration of FSDP workers without Ray.
    
    This maintains the same interface as verl's Ray-based orchestrator.
    """
    
    def __init__(self, config):
        self.config = config
        self.workers = {}
        
        # Initialize workers based on config
        self._init_workers()
    
    def _init_workers(self):
        """Initialize all required workers locally."""
        logger.info("Initializing local FSDP workers...")
        
        # Actor worker (always needed)
        if hasattr(self.config, 'actor_rollout_ref'):
            self.workers['actor'] = LocalFSDPActorWorker(self.config.actor_rollout_ref)
            logger.info("Initialized local FSDP actor worker")
        
        # Critic worker (if needed)
        if hasattr(self.config, 'critic') and self.config.critic is not None:
            self.workers['critic'] = LocalFSDPCriticWorker(self.config.critic)
            logger.info("Initialized local FSDP critic worker")
    
    def get_worker(self, role: str):
        """Get a worker by role."""
        return self.workers.get(role)
    
    def generate_sequences(self, prompts: DataProto) -> DataProto:
        """Generate sequences using the actor worker."""
        return self.workers['actor'].generate_sequences(prompts)
    
    def compute_values(self, data: DataProto) -> Optional[DataProto]:
        """Compute critic values if critic is available."""
        if 'critic' in self.workers:
            return self.workers['critic'].compute_values(data)
        return None
    
    def update_actor(self, data: DataProto) -> Dict[str, Any]:
        """Update actor policy with PPO mini-batch training."""
        if 'actor' in self.workers:
            return self.workers['actor'].update_policy(data)
        else:
            raise ValueError("Actor worker not found")
    
    def update_critic(self, data: DataProto) -> Dict[str, Any]:
        """Update critic with data."""
        if 'critic' in self.workers:
            return self.workers['critic'].train_step(data)
        else:
            raise ValueError("Critic worker not found")
    
    def train_step(self, data: DataProto, role: str = 'actor') -> Dict[str, Any]:
        """Legacy method for training step on specified worker."""
        if role == 'actor':
            return self.update_actor(data)
        elif role == 'critic':
            return self.update_critic(data)
        else:
            raise ValueError(f"Worker role '{role}' not found")
    
    def save_checkpoint(self, checkpoint_path: str):
        """Save checkpoints for all workers."""
        logger.info(f"Saving local FSDP worker checkpoints to {checkpoint_path}")
        
        for role, worker in self.workers.items():
            worker_path = os.path.join(checkpoint_path, role)
            worker.save_checkpoint(worker_path)
    
    def load_checkpoint(self, checkpoint_path: str):
        """Load checkpoints for all workers."""
        logger.info(f"Loading local FSDP worker checkpoints from {checkpoint_path}")
        
        for role, worker in self.workers.items():
            worker_path = os.path.join(checkpoint_path, role)
            if os.path.exists(worker_path):
                worker.load_checkpoint(worker_path)
    
    def cleanup(self):
        """Clean up resources."""
        logger.info("Cleaning up local FSDP workers")
        # FSDP cleanup if needed


# Convenience functions for API compatibility
def create_local_fsdp_workers(config) -> LocalFSDPWorkers:
    """Create local FSDP workers with the same API as Ray version."""
    return LocalFSDPWorkers(config)