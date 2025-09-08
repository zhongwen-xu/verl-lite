#!/usr/bin/env python3
# Copyright 2025 verl-lite Authors
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
Example demonstrating weight synchronization between trainer and rollout server.

This shows how the PPO trainer automatically syncs updated actor weights
to the rollout server during RL training.
"""

import logging
import torch
from pathlib import Path

# Import verl-lite components
from verl_lite.trainer.ppo_trainer_local import LocalPPOTrainer, LocalTrainingConfig
from verl_lite.workers.rollout_local import LocalRolloutManager, VerlStyleWeightSync
from verl import DataProto

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def create_sample_config():
    """Create a sample training configuration."""
    from omegaconf import OmegaConf
    from dataclasses import dataclass
    
    # Mock config structure similar to verl
    @dataclass
    class MockConfig:
        model_path: str = "gpt2"
        
    @dataclass  
    class MockRolloutConfig:
        model_path: str = "gpt2"
        response_length: int = 128
        temperature: float = 1.0
        max_length: int = 512
        
    @dataclass
    class MockAlgoConfig:
        gamma: float = 0.99
        lam: float = 0.95
        clip_ratio: float = 0.2
        
    @dataclass
    class MockActorConfig:
        model: MockConfig = MockConfig()
        rollout: MockRolloutConfig = MockRolloutConfig()

    config = LocalTrainingConfig(
        actor_rollout_ref=MockActorConfig(),
        total_epochs=2,
        rollout_engine="vllm",
        sync_weights_frequency=1,  # Sync every step
        enable_weight_sync=True,
        checkpoint_dir="./weight_sync_checkpoints",
        log_dir="./weight_sync_logs"
    )
    
    return config


def create_sample_data():
    """Create sample training data."""
    import numpy as np
    from tensordict import TensorDict
    
    batch_size = 4
    seq_len = 64
    
    # Create sample batch
    batch = TensorDict({
        "input_ids": torch.randint(0, 1000, (batch_size, seq_len)),
        "attention_mask": torch.ones(batch_size, seq_len),
        "response_ids": torch.randint(0, 1000, (batch_size, 32)),
        "response_mask": torch.ones(batch_size, 32),
    })
    
    # Create sample non-tensor data
    non_tensor_batch = {
        "prompt_text": np.array([f"Sample prompt {i}" for i in range(batch_size)], dtype=object),
        "response_text": np.array([f"Sample response {i}" for i in range(batch_size)], dtype=object),
        "rewards": np.array([0.5, 0.8, 0.3, 0.9]),
        "advantages": np.array([0.1, 0.4, -0.2, 0.6]),
    }
    
    return DataProto(
        batch=batch,
        non_tensor_batch=non_tensor_batch,
        meta_info={"sample": True}
    )


def demonstrate_weight_sync():
    """Demonstrate weight synchronization functionality."""
    logger.info("Starting weight synchronization demonstration")
    
    try:
        # Create configuration
        config = create_sample_config()
        
        # Create trainer with weight sync enabled
        trainer = LocalPPOTrainer(config)
        
        # Check if weight sync is properly initialized
        if trainer.weight_sync_manager:
            logger.info("✓ Weight sync manager initialized")
            logger.info(f"✓ Sync frequency: {trainer.weight_sync_manager.sync_frequency}")
        else:
            logger.warning("✗ Weight sync manager not initialized")
            return
            
        # Get rollout model info before training
        logger.info("Rollout model info before training:")
        model_info = trainer.get_rollout_model_info()
        logger.info(f"  Model version: {model_info.get('model_version', 'N/A')}")
        logger.info(f"  Model path: {model_info.get('current_model_path', 'N/A')}")
        
        # Create sample data
        sample_data = create_sample_data()
        
        # Demonstrate rollout with current weights
        logger.info("Performing initial rollout...")
        responses = trainer.rollout(sample_data)
        logger.info(f"✓ Generated {len(responses)} responses")
        
        # Demonstrate weight sync during training step
        logger.info("Performing training step with weight sync...")
        step_metrics = trainer.ppo_step(sample_data)
        
        # Check sync status
        sync_success = step_metrics.get('weight_sync_success', False)
        if sync_success:
            logger.info("✓ Weight sync successful")
        else:
            logger.warning("✗ Weight sync failed")
            
        # Get updated model info
        logger.info("Rollout model info after training:")
        updated_model_info = trainer.get_rollout_model_info()
        logger.info(f"  Model version: {updated_model_info.get('model_version', 'N/A')}")
        logger.info(f"  Model path: {updated_model_info.get('current_model_path', 'N/A')}")
        
        # Demonstrate forced weight sync
        logger.info("Demonstrating forced weight sync...")
        force_sync_success = trainer.force_weight_sync()
        if force_sync_success:
            logger.info("✓ Forced weight sync successful")
        else:
            logger.warning("✗ Forced weight sync failed")
            
        # Demonstrate weight sync validation
        logger.info("Validating weight synchronization...")
        is_synced = trainer.validate_weight_sync()
        if is_synced:
            logger.info("✓ Weight sync validation passed")
        else:
            logger.warning("✗ Weight sync validation failed")
            
        logger.info("Weight synchronization demonstration completed successfully")
        
    except Exception as e:
        logger.error(f"Error during weight sync demonstration: {e}")
        raise
    finally:
        # Cleanup
        if 'trainer' in locals():
            trainer.workers.cleanup()
            if trainer.rollout_manager:
                trainer.rollout_manager.stop()


def demonstrate_standalone_weight_sync():
    """Demonstrate standalone weight synchronization without full trainer."""
    logger.info("Starting standalone weight sync demonstration")
    
    try:
        # Create a mock rollout config
        from dataclasses import dataclass
        
        @dataclass
        class MockRolloutConfig:
            model_path: str = "gpt2"
            tensor_parallel_size: int = 1
            max_length: int = 512
            
        # Create rollout manager
        rollout_config = MockRolloutConfig()
        rollout_manager = LocalRolloutManager(
            rollout_config, 
            engine_type="vllm",
            checkpoint_dir="./standalone_checkpoints"
        )
        
        # Create weight sync manager using verl pattern
        weight_sync = VerlStyleWeightSync(rollout_manager)
        
        # Start rollout server
        logger.info("Starting rollout server...")
        rollout_manager.start()
        
        # Create mock model with some weights
        import torch.nn as nn
        mock_model = nn.Linear(100, 50)
        
        # Demonstrate direct weight sync using verl pattern
        logger.info("Syncing mock model weights...")
        import asyncio
        loop = asyncio.get_event_loop()
        success = loop.run_until_complete(weight_sync.sync_weights_from_fsdp_model(mock_model))
        
        if success:
            logger.info("✓ Standalone weight sync successful")
        else:
            logger.warning("✗ Standalone weight sync failed")
            
        # Get model info
        model_info = rollout_manager.get_model_info()
        logger.info(f"Model info: {model_info}")
        
        logger.info("Standalone weight sync demonstration completed")
        
    except Exception as e:
        logger.error(f"Error in standalone weight sync: {e}")
        raise
    finally:
        # Cleanup
        if 'rollout_manager' in locals():
            rollout_manager.stop()


if __name__ == "__main__":
    print("=" * 60)
    print("verl-lite Weight Synchronization Demonstration")
    print("=" * 60)
    
    # Run demonstrations
    try:
        print("\n1. Full Trainer Weight Sync Demo")
        print("-" * 40)
        demonstrate_weight_sync()
        
        print("\n2. Standalone Weight Sync Demo")
        print("-" * 40)
        demonstrate_standalone_weight_sync()
        
    except Exception as e:
        logger.error(f"Demonstration failed: {e}")
        exit(1)
    
    print("\n" + "=" * 60)
    print("All demonstrations completed!")
    print("=" * 60)