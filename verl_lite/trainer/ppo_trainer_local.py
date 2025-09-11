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
Local PPO trainer without Ray dependencies for verl-lite.

This trainer maintains the same API as the original Ray-based PPO trainer
but orchestrates everything locally on a single machine.
"""

import logging
import os
from typing import Dict, Any, Optional, Union
from dataclasses import dataclass

# Import core components from verl
try:
    from verl_lite import TensorDict, tu
    from verl.trainer.ppo import core_algos
    from verl.trainer.ppo.core_algos import AdvantageEstimator, agg_loss
    from verl.trainer.ppo.reward import compute_reward
    from verl.trainer.config import AlgoConfig
    from verl.utils.tracking import ValidationGenerationsLogger
    from verl.utils.metric import reduce_metrics
    from verl.trainer.ppo.utils import need_critic, need_reference_policy, need_reward_model
    from verl.utils.config import omega_conf_to_dataclass
    from omegaconf import DictConfig, OmegaConf
except ImportError as e:
    raise ImportError(f"Cannot import verl trainer components: {e}")

# Import our local components
from ..workers.fsdp_workers_local import LocalFSDPWorkers
from ..workers.rollout_local import LocalRolloutManager, WeightSyncManager

logger = logging.getLogger(__name__)


@dataclass
class LocalTrainingConfig:
    """Configuration for local training without Ray."""
    
    # Model configs (import directly from verl configs)
    actor_rollout_ref: Any = None
    critic: Any = None
    reward_model: Any = None
    
    # Algorithm config  
    algo: AlgoConfig = None
    
    # Rollout config
    rollout_engine: str = "vllm"  # "vllm" or "sglang"
    
    # Training settings
    total_epochs: int = 1
    save_freq: int = 1
    log_freq: int = 1
    validate_freq: int = 10
    
    # Weight synchronization settings
    sync_weights_frequency: int = 1  # Sync every N steps
    enable_weight_sync: bool = True
    
    # Local settings
    checkpoint_dir: str = "./checkpoints"
    log_dir: str = "./logs"
    
    @classmethod
    def from_verl_config(cls, config: Union[DictConfig, dict]) -> 'LocalTrainingConfig':
        """Create LocalTrainingConfig from a verl config (DictConfig or dict)."""
        if isinstance(config, dict):
            config = OmegaConf.create(config)
        
        # Extract algorithm config
        algo_config = None
        if 'algorithm' in config:
            algo_config = omega_conf_to_dataclass(config.algorithm, AlgoConfig)
        
        # Extract training settings from trainer section
        trainer_config = config.get('trainer', {})
        total_epochs = trainer_config.get('total_epochs', 1)
        save_freq = trainer_config.get('save_freq', 1)
        
        # Extract checkpoint and log directories
        checkpoint_dir = trainer_config.get('default_local_dir', './checkpoints')
        log_dir = './logs'  # Default log directory
        
        # Extract weight sync settings
        sync_frequency = trainer_config.get('sync_weights_frequency', 1)
        enable_sync = trainer_config.get('enable_weight_sync', True)
        
        # Create instance with verl config sections
        return cls(
            actor_rollout_ref=config.get('actor_rollout_ref'),
            critic=config.get('critic'),
            reward_model=config.get('reward_model'),
            algo=algo_config,
            rollout_engine="vllm",  # Default, can be overridden
            total_epochs=total_epochs,
            save_freq=save_freq,
            sync_weights_frequency=sync_frequency,
            enable_weight_sync=enable_sync,
            checkpoint_dir=checkpoint_dir,
            log_dir=log_dir
        )


class LocalPPOTrainer:
    """
    Local PPO trainer that maintains the same API as the Ray-based trainer.
    
    This trainer orchestrates PPO training locally without Ray, making it
    suitable for debugging and prototyping on single machines.
    """
    
    def __init__(self, config: LocalTrainingConfig):
        self.config = config
        
        # Initialize logging
        self._setup_logging()
        
        # Initialize workers
        self._init_workers()
        
        # Initialize algorithm components
        self._init_algorithm()
        
        # Initialize tracking
        self._init_tracking()
        
        logger.info("Local PPO trainer initialized")
    
    def _setup_logging(self):
        """Set up logging."""
        os.makedirs(self.config.log_dir, exist_ok=True)
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
    
    def _init_workers(self):
        """Initialize local workers."""
        logger.info("Initializing local workers...")
        
        # Initialize FSDP workers
        self.workers = LocalFSDPWorkers(self.config)
        
        # Initialize rollout manager with weight sync support
        if hasattr(self.config, 'actor_rollout_ref') and self.config.actor_rollout_ref:
            rollout_config = self.config.actor_rollout_ref.rollout
            self.rollout_manager = LocalRolloutManager(
                rollout_config, 
                self.config.rollout_engine,
                checkpoint_dir=self.config.checkpoint_dir
            )
            
            # Initialize weight synchronization
            if self.config.enable_weight_sync:
                self.weight_sync_manager = WeightSyncManager(self.rollout_manager)
                self.weight_sync_manager.set_sync_frequency(self.config.sync_weights_frequency)
                logger.info(f"Initialized weight sync with frequency {self.config.sync_weights_frequency}")
            else:
                self.weight_sync_manager = None
                
            logger.info(f"Initialized {self.config.rollout_engine} rollout manager")
        else:
            self.rollout_manager = None
            self.weight_sync_manager = None
            logger.info("No rollout manager configured")
    
    def _init_algorithm(self):
        """Initialize algorithm components."""
        if self.config.algo:
            self.advantage_estimator = AdvantageEstimator(
                gamma=self.config.algo.gamma,
                lam=self.config.algo.lam
            )
            logger.info("Initialized advantage estimator")
        else:
            self.advantage_estimator = None
    
    def _init_tracking(self):
        """Initialize tracking and validation."""
        self.validation_logger = ValidationGenerationsLogger()
        self.global_step = 0
        self.epoch = 0
    
    def force_weight_sync(self) -> bool:
        """Force immediate weight synchronization to rollout server."""
        if self.weight_sync_manager and 'actor' in self.workers.workers:
            try:
                actor_model = self.workers.get_model('actor')
                return self.weight_sync_manager.force_sync(actor_model, self.global_step)
            except Exception as e:
                logger.error(f"Failed to force weight sync: {e}")
                return False
        return True  # No sync needed
    
    def get_rollout_model_info(self) -> Dict[str, Any]:
        """Get information about the current rollout model."""
        if self.rollout_manager:
            return self.rollout_manager.get_model_info()
        return {}
    
    def validate_weight_sync(self) -> bool:
        """Validate that rollout server has the latest weights."""
        if not self.weight_sync_manager:
            return True  # No validation needed
            
        try:
            model_info = self.get_rollout_model_info()
            current_version = model_info.get('model_version', 0)
            expected_version = self.global_step // self.config.sync_weights_frequency
            
            is_synced = current_version >= expected_version
            if not is_synced:
                logger.warning(
                    f"Weight sync validation failed: rollout model version {current_version}, "
                    f"expected >= {expected_version}"
                )
            return is_synced
        except Exception as e:
            logger.error(f"Weight sync validation error: {e}")
            return False
    
    def rollout(self, prompts: TensorDict) -> TensorDict:
        """
        Perform rollout to generate responses.
        
        This maintains the same API as the Ray trainer but runs locally.
        """
        logger.info(f"Starting rollout for {len(prompts)} prompts")
        
        if self.rollout_manager:
            # Use local rollout server
            responses = self.rollout_manager.generate_sequences(prompts)
        else:
            # Use actor worker directly
            responses = self.workers.generate_sequences(prompts)
        
        logger.info(f"Rollout completed, generated {len(responses)} responses")
        return responses
    
    def compute_values(self, data: TensorDict) -> Optional[TensorDict]:
        """Compute critic values if critic is available."""
        if need_critic(self.config.algo):
            values = self.workers.compute_values(data)
            logger.info("Computed critic values")
            return values
        return None
    
    def compute_rewards(self, data: TensorDict) -> TensorDict:
        """Compute rewards for the generated responses."""
        if need_reward_model(self.config.algo):
            # Use reward model worker
            rewards = self.workers.compute_rewards(data)
        else:
            # Use external reward function
            rewards = compute_reward(data, self.config)
        
        logger.info("Computed rewards")
        return rewards
    
    def compute_advantages(self, values: TensorDict, rewards: TensorDict) -> TensorDict:
        """Compute advantages using GAE."""
        if self.advantage_estimator:
            advantages = self.advantage_estimator.estimate(values, rewards)
            logger.info("Computed advantages")
            return advantages
        else:
            # Simple advantage = reward
            advantages = rewards
            logger.info("Using simple advantages (reward)")
            return advantages
    
    def ppo_step(self, data: TensorDict) -> Dict[str, Any]:
        """
        Perform a PPO training step.
        
        This is the core RL training loop step.
        """
        logger.info("Starting PPO step")
        
        # Train actor
        actor_metrics = self.workers.train_step(data, role='actor')
        
        # Synchronize actor weights to rollout server after training
        sync_success = True
        if self.weight_sync_manager and 'actor' in self.workers.workers:
            try:
                actor_model = self.workers.get_model('actor')
                sync_success = self.weight_sync_manager.sync_weights_from_trainer(
                    actor_model, self.global_step
                )
                if not sync_success:
                    logger.warning(f"Failed to sync weights at step {self.global_step}")
            except Exception as e:
                logger.error(f"Error during weight sync: {e}")
                sync_success = False
        
        # Train critic if available
        critic_metrics = {}
        if 'critic' in self.workers.workers:
            critic_metrics = self.workers.train_step(data, role='critic')
        
        # Combine metrics
        metrics = {
            **{f"actor/{k}": v for k, v in actor_metrics.items()},
            **{f"critic/{k}": v for k, v in critic_metrics.items()},
            "weight_sync_success": sync_success,
        }
        
        if self.weight_sync_manager:
            model_info = self.rollout_manager.get_model_info()
            metrics["rollout_model_version"] = model_info.get('model_version', 0)
        
        logger.info("PPO step completed")
        return metrics
    
    def train_epoch(self, dataset) -> Dict[str, Any]:
        """
        Train for one epoch.
        
        This is the main training loop that combines rollout, reward computation,
        and PPO updates.
        """
        logger.info(f"Starting epoch {self.epoch}")
        
        epoch_metrics = {}
        
        for batch_idx, prompts in enumerate(dataset):
            logger.info(f"Processing batch {batch_idx}")
            
            # 1. Rollout - generate responses
            responses = self.rollout(prompts)
            
            # 2. Compute values (if critic available)
            values = self.compute_values(responses)
            
            # 3. Compute rewards
            rewards = self.compute_rewards(responses)
            
            # 4. Compute advantages  
            if values is not None:
                advantages = self.compute_advantages(values, rewards)
            else:
                advantages = rewards
            
            # 5. Prepare training data - merge TensorDicts
            train_data = tu.union_tensor_dict(responses, rewards)
            if advantages is not None:
                train_data = tu.union_tensor_dict(train_data, advantages)
            
            # 6. PPO training step
            step_metrics = self.ppo_step(train_data)
            
            # 7. Update global step
            self.global_step += 1
            
            # 8. Validate weight sync periodically
            if (self.weight_sync_manager and 
                self.global_step % (self.config.sync_weights_frequency * 10) == 0):
                if not self.validate_weight_sync():
                    logger.warning("Weight sync validation failed, forcing sync")
                    self.force_weight_sync()
            
            # 9. Log metrics
            if self.global_step % self.config.log_freq == 0:
                logger.info(f"Step {self.global_step} metrics: {step_metrics}")
                
            # 10. Accumulate epoch metrics
            for key, value in step_metrics.items():
                if key not in epoch_metrics:
                    epoch_metrics[key] = []
                epoch_metrics[key].append(value)
        
        # Average epoch metrics
        avg_metrics = {}
        for key, values in epoch_metrics.items():
            if values:  # Check if list is not empty
                avg_metrics[f"epoch_{key}"] = sum(values) / len(values)
        
        logger.info(f"Epoch {self.epoch} completed with metrics: {avg_metrics}")
        return avg_metrics
    
    def fit(self, train_dataset) -> Dict[str, Any]:
        """
        Main training loop.
        
        This maintains the same API as the Ray trainer's fit method.
        """
        logger.info("Starting training...")
        
        # Start rollout manager if configured
        if self.rollout_manager:
            self.rollout_manager.start()
            
            # Initial weight sync to ensure rollout server has latest actor weights
            if self.weight_sync_manager and 'actor' in self.workers.workers:
                logger.info("Performing initial weight sync to rollout server")
                success = self.force_weight_sync()
                if not success:
                    logger.warning("Initial weight sync failed, continuing with training")
        
        try:
            training_metrics = {}
            
            for epoch in range(self.config.total_epochs):
                self.epoch = epoch
                
                # Train one epoch
                epoch_metrics = self.train_epoch(train_dataset)
                
                # Save checkpoint
                if epoch % self.config.save_freq == 0:
                    checkpoint_path = os.path.join(self.config.checkpoint_dir, f"epoch_{epoch}")
                    self.save_checkpoint(checkpoint_path)
                
                # Accumulate training metrics
                for key, value in epoch_metrics.items():
                    if key not in training_metrics:
                        training_metrics[key] = []
                    training_metrics[key].append(value)
            
            logger.info("Training completed successfully")
            return training_metrics
            
        finally:
            # Clean up
            if self.rollout_manager:
                # Exit rollout mode before stopping
                if self.weight_sync_manager:
                    import asyncio
                    loop = asyncio.get_event_loop()
                    loop.run_until_complete(self.weight_sync_manager.exit_rollout_mode())
                self.rollout_manager.stop()
            self.workers.cleanup()
    
    def save_checkpoint(self, checkpoint_path: str):
        """Save training checkpoint."""
        os.makedirs(checkpoint_path, exist_ok=True)
        
        # Save worker checkpoints
        self.workers.save_checkpoint(checkpoint_path)
        
        # Save training state
        training_state = {
            "global_step": self.global_step,
            "epoch": self.epoch,
        }
        
        import json
        with open(os.path.join(checkpoint_path, "training_state.json"), "w") as f:
            json.dump(training_state, f)
        
        logger.info(f"Checkpoint saved to {checkpoint_path}")
    
    def load_checkpoint(self, checkpoint_path: str):
        """Load training checkpoint.""" 
        # Load worker checkpoints
        self.workers.load_checkpoint(checkpoint_path)
        
        # Load training state
        training_state_path = os.path.join(checkpoint_path, "training_state.json")
        if os.path.exists(training_state_path):
            import json
            with open(training_state_path, "r") as f:
                training_state = json.load(f)
            
            self.global_step = training_state["global_step"]
            self.epoch = training_state["epoch"]
        
        # Force weight sync after loading checkpoint
        if self.weight_sync_manager:
            logger.info("Syncing weights to rollout server after checkpoint load")
            success = self.force_weight_sync()
            if not success:
                logger.warning("Weight sync after checkpoint load failed")
        
        logger.info(f"Checkpoint loaded from {checkpoint_path}")


# Convenience function
def create_local_ppo_trainer(config: LocalTrainingConfig) -> LocalPPOTrainer:
    """Create a local PPO trainer."""
    return LocalPPOTrainer(config)