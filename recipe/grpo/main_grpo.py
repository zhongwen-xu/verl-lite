#!/usr/bin/env python3
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
GRPO training example for verl-lite.

This demonstrates how to implement GRPO (Group Relative Policy Optimization) 
using verl-lite components. The code structure is designed to be easily 
migratable to full verl with Ray.
"""

import os
import argparse
import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

# Import verl components directly (maintaining same API)
try:
    from verl import DataProto
    from verl.trainer.config import AlgoConfig
    from verl.workers.config import FSDPActorConfig, RolloutConfig
    from verl.utils.config import omega_conf_to_dataclass
    import verl.utils.hdfs_io as hdfs_io
except ImportError as e:
    raise ImportError(f"Cannot import verl components. Make sure verl is installed: {e}")

# Import verl-lite local components
from verl_lite.trainer import LocalPPOTrainer
from verl_lite.trainer.ppo_trainer_local import LocalTrainingConfig
from verl_lite.workers import LocalFSDPWorkers, LocalRolloutManager

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class GRPOConfig(LocalTrainingConfig):
    """GRPO-specific configuration."""
    
    # GRPO algorithm settings
    grpo_gamma: float = 1.0
    grpo_lambda: float = 0.95
    grpo_epsilon: float = 0.2
    grpo_entropy_coeff: float = 0.01
    grpo_vf_coeff: float = 0.5
    
    # Data settings
    train_files: str = ""
    val_files: str = ""
    max_prompt_length: int = 512
    max_response_length: int = 512
    train_batch_size: int = 32
    
    # Model settings
    model_path: str = ""
    use_flash_attention: bool = True
    
    # Reward function
    reward_function_path: str = ""
    reward_function_name: str = ""


class SimpleGRPODataset:
    """Simple dataset for GRPO training."""
    
    def __init__(self, data_files: List[str], tokenizer, max_prompt_length: int = 512):
        self.data_files = data_files
        self.tokenizer = tokenizer
        self.max_prompt_length = max_prompt_length
        self.data = []
        
        self._load_data()
    
    def _load_data(self):
        """Load data from parquet files."""
        import pandas as pd
        
        for file_path in self.data_files:
            if file_path.endswith('.parquet'):
                df = pd.read_parquet(file_path)
                for _, row in df.iterrows():
                    self.data.append({
                        'prompt': row.get('prompt', ''),
                        'ground_truth': row.get('ground_truth', ''),
                    })
        
        logger.info(f"Loaded {len(self.data)} examples")
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        """Get a batch of prompts as DataProto."""
        if isinstance(idx, slice):
            # Handle slicing
            start, stop, step = idx.indices(len(self.data))
            batch_data = self.data[start:stop:step]
        elif isinstance(idx, int):
            # Single item
            batch_data = [self.data[idx]]
        else:
            # List of indices
            batch_data = [self.data[i] for i in idx]
        
        # Convert to DataProto
        prompts = [item['prompt'] for item in batch_data]
        ground_truths = [item['ground_truth'] for item in batch_data]
        
        # Tokenize prompts
        tokenized = self.tokenizer(
            prompts,
            max_length=self.max_prompt_length,
            padding=True,
            truncation=True,
            return_tensors="pt"
        )
        
        from tensordict import TensorDict
        import numpy as np
        
        batch_tensor = TensorDict({
            "input_ids": tokenized["input_ids"],
            "attention_mask": tokenized["attention_mask"],
        }, batch_size=(len(prompts),))
        
        non_tensor_batch = {
            "prompts": np.array(prompts, dtype=object),
            "ground_truths": np.array(ground_truths, dtype=object),
        }
        
        return DataProto(
            batch=batch_tensor,
            non_tensor_batch=non_tensor_batch,
            meta_info={"batch_size": len(prompts)}
        )
    
    def get_dataloader(self, batch_size: int):
        """Create a simple dataloader."""
        for i in range(0, len(self.data), batch_size):
            yield self[i:i+batch_size]


def load_reward_function(reward_function_path: str, reward_function_name: str):
    """Load custom reward function."""
    import importlib.util
    
    spec = importlib.util.spec_from_file_location("reward_module", reward_function_path)
    reward_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(reward_module)
    
    return getattr(reward_module, reward_function_name)


def create_grpo_trainer(config: GRPOConfig) -> LocalPPOTrainer:
    """Create a GRPO trainer with verl-lite components."""
    
    # Configure algorithm for GRPO
    algo_config = AlgoConfig(
        adv_estimator="grpo",  # Use GRPO advantage estimator
        gamma=config.grpo_gamma,
        lam=config.grpo_lambda,
        ppo_eps=config.grpo_epsilon,
        entropy_coeff=config.grpo_entropy_coeff,
        use_kl_in_reward=False,  # GRPO doesn't use KL in reward
    )
    config.algo = algo_config
    
    # Create trainer
    trainer = LocalPPOTrainer(config)
    
    return trainer


class GRPOTrainer(LocalPPOTrainer):
    """
    GRPO-specific trainer that extends LocalPPOTrainer.
    
    This maintains the same structure as verl GRPO but runs locally.
    """
    
    def __init__(self, config: GRPOConfig):
        super().__init__(config)
        
        # Load custom reward function if specified
        self.reward_function = None
        if config.reward_function_path and config.reward_function_name:
            self.reward_function = load_reward_function(
                config.reward_function_path,
                config.reward_function_name
            )
            logger.info(f"Loaded reward function: {config.reward_function_name}")
    
    def compute_rewards(self, data: DataProto) -> DataProto:
        """Compute rewards using custom reward function."""
        if self.reward_function:
            # Use custom reward function
            rewards = []
            
            prompts = data.non_tensor_batch.get("prompts", [])
            responses = data.non_tensor_batch.get("response_text", [])
            ground_truths = data.non_tensor_batch.get("ground_truths", [])
            
            for prompt, response, gt in zip(prompts, responses, ground_truths):
                reward = self.reward_function(
                    data_source="custom",
                    solution_str=response,
                    ground_truth=gt,
                    extra_info={"prompt": prompt}
                )
                rewards.append(reward)
            
            import numpy as np
            reward_array = np.array(rewards, dtype=np.float32)
            
            # Create reward DataProto
            reward_data = DataProto(
                batch=data.batch.clone() if data.batch else None,
                non_tensor_batch={
                    **data.non_tensor_batch,
                    "rewards": reward_array
                },
                meta_info=data.meta_info
            )
            
            logger.info(f"Computed {len(rewards)} rewards, avg: {np.mean(rewards):.4f}")
            return reward_data
        else:
            # Fall back to parent implementation
            return super().compute_rewards(data)


def main():
    parser = argparse.ArgumentParser(description="GRPO training with verl-lite")
    parser.add_argument("--config", type=str, help="Config file (optional)")
    parser.add_argument("--model_path", type=str, required=True, help="Path to the model")
    parser.add_argument("--train_files", type=str, required=True, help="Training data files")
    parser.add_argument("--val_files", type=str, help="Validation data files")
    parser.add_argument("--reward_function_path", type=str, help="Custom reward function file")
    parser.add_argument("--reward_function_name", type=str, default="reward_function", help="Reward function name")
    parser.add_argument("--rollout_engine", type=str, default="vllm", choices=["vllm", "sglang"], help="Rollout engine")
    parser.add_argument("--total_epochs", type=int, default=2, help="Total training epochs")
    parser.add_argument("--batch_size", type=int, default=32, help="Training batch size")
    parser.add_argument("--output_dir", type=str, default="./grpo_output", help="Output directory")
    
    args = parser.parse_args()
    
    # Load config from verl config file if provided
    if args.config:
        from omegaconf import OmegaConf
        import hydra
        from hydra import initialize_config_dir, compose
        from pathlib import Path
        
        logger.info(f"Loading verl config from: {args.config}")
        config_path = Path(args.config).resolve()
        config_dir = str(config_path.parent)
        config_name = config_path.stem
        
        # Initialize Hydra and load config
        with initialize_config_dir(config_dir=config_dir, version_base=None):
            verl_config = compose(config_name=config_name)
        
        # Convert to local config format
        base_config = LocalTrainingConfig.from_verl_config(verl_config)
        
        # Create GRPO config with verl config base
        config = GRPOConfig(
            **base_config.__dict__,
            train_files=args.train_files or "",
            val_files=args.val_files or "",
            model_path=args.model_path or "",
            reward_function_path=args.reward_function_path or "",
            reward_function_name=args.reward_function_name,
        )
        
        # Extract additional settings from verl config
        if hasattr(verl_config, 'actor_rollout_ref') and 'model' in verl_config.actor_rollout_ref:
            if not args.model_path:  # Only use from config if not overridden
                config.model_path = verl_config.actor_rollout_ref.model.get('path', '')
                
        logger.info("Successfully loaded and converted verl config")
    else:
        # Create config from command line args (original behavior)  
        if not args.model_path or not args.train_files:
            parser.error("--model_path and --train_files are required when --config is not provided")
            
        config = GRPOConfig(
            model_path=args.model_path,
            train_files=args.train_files,
            val_files=args.val_files or "",
            total_epochs=args.total_epochs,
            rollout_engine=args.rollout_engine,
            reward_function_path=args.reward_function_path or "",
            reward_function_name=args.reward_function_name,
            checkpoint_dir=os.path.join(args.output_dir, "checkpoints"),
            log_dir=os.path.join(args.output_dir, "logs"),
        )
    
    logger.info("Starting GRPO training with verl-lite")
    logger.info(f"Model: {config.model_path}")
    logger.info(f"Train files: {config.train_files}")
    logger.info(f"Rollout engine: {config.rollout_engine}")
    
    # Load tokenizer
    from verl.utils.tokenizer import HFTokenizer
    tokenizer = HFTokenizer(config.model_path)
    
    # Create dataset
    train_files = [f.strip() for f in config.train_files.split(",")]
    train_dataset = SimpleGRPODataset(train_files, tokenizer.tokenizer, config.max_prompt_length)
    
    # Create trainer
    trainer = GRPOTrainer(config)
    
    # Start training
    try:
        results = trainer.fit(train_dataset.get_dataloader(config.train_batch_size))
        logger.info("Training completed successfully!")
        logger.info(f"Results: {results}")
        
    except Exception as e:
        logger.error(f"Training failed: {e}")
        import traceback
        traceback.print_exc()
        
    logger.info("GRPO training finished")


if __name__ == "__main__":
    main()