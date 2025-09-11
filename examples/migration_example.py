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
Migration example showing how the same code works in both verl-lite and full verl.

This demonstrates the key principle: write once, migrate easily.
"""

import os
from typing import Dict, Any

# Configuration flag to switch between local and distributed
USE_RAY = os.getenv("USE_RAY", "False").lower() == "true"

# Import core components (same for both)
from verl_lite import TensorDict, tu
from verl.trainer.config import AlgoConfig

if USE_RAY:
    print("=== Using Full VERL with Ray ===")
    
    # Full verl imports
    import ray
    from verl.trainer.ppo.ray_trainer import RayPPOTrainer
    from verl.workers.fsdp_workers import ActorRolloutRefWorker, CriticWorker
    
    # Initialize Ray
    ray.init(ignore_reinit_error=True)
    
    def create_trainer(config):
        """Create Ray-based trainer."""
        return RayPPOTrainer(config)
    
    def cleanup():
        """Cleanup Ray resources."""
        ray.shutdown()
        
else:
    print("=== Using verl-lite (Local Mode) ===")
    
    # verl-lite imports
    from verl_lite.trainer import LocalPPOTrainer
    from verl_lite.workers import LocalFSDPWorkers
    
    def create_trainer(config):
        """Create local trainer.""" 
        return LocalPPOTrainer(config)
    
    def cleanup():
        """Cleanup local resources."""
        print("Local cleanup complete")


class AdaptiveTrainingPipeline:
    """
    Training pipeline that works with both verl-lite and full verl.
    
    This demonstrates how to write code that's portable between both versions.
    """
    
    def __init__(self, config):
        self.config = config
        self.trainer = create_trainer(config)
        
    def train(self, dataset) -> Dict[str, Any]:
        """
        Main training method - identical for both versions.
        
        The beauty is that this method doesn't need to change when migrating!
        """
        print(f"Starting training with {'Ray' if USE_RAY else 'local'} backend")
        
        try:
            # Training loop (same for both versions)
            results = self.trainer.fit(dataset)
            
            print("Training completed successfully!")
            return results
            
        except Exception as e:
            print(f"Training failed: {e}")
            raise
        
        finally:
            cleanup()
    
    def save_checkpoint(self, path: str):
        """Save checkpoint - same API for both versions."""
        self.trainer.save_checkpoint(path)
        print(f"Checkpoint saved to {path}")
    
    def load_checkpoint(self, path: str):
        """Load checkpoint - same API for both versions."""
        self.trainer.load_checkpoint(path)
        print(f"Checkpoint loaded from {path}")


def create_sample_config():
    """Create a sample configuration that works for both versions."""
    from dataclasses import dataclass
    
    @dataclass
    class SampleConfig:
        # Algorithm config (same for both)
        algo: AlgoConfig = None
        
        # Model path
        model_path: str = "microsoft/DialoGPT-small"
        
        # Training params
        total_epochs: int = 2
        batch_size: int = 4
        
        # Output
        checkpoint_dir: str = "./checkpoints"
        log_dir: str = "./logs"
    
    config = SampleConfig()
    config.algo = AlgoConfig(
        adv_estimator="grpo",
        gamma=0.99,
        lam=0.95
    )
    
    return config


def demonstrate_migration():
    """Demonstrate how the same code works in both environments."""
    
    print("\n" + "="*60)
    print("MIGRATION DEMONSTRATION")
    print("="*60)
    
    # Create config (identical for both)
    config = create_sample_config()
    print(f"✅ Config created: {config.model_path}")
    
    # Create trainer (different implementation, same API)
    pipeline = AdaptiveTrainingPipeline(config)
    print(f"✅ Pipeline created with {'Ray' if USE_RAY else 'local'} backend")
    
    # Demonstrate identical APIs
    print("\n--- API Demonstration ---")
    
    # These method calls are IDENTICAL regardless of backend
    try:
        pipeline.save_checkpoint("./demo_checkpoint")
        print("✅ Checkpoint save API works")
        
        pipeline.load_checkpoint("./demo_checkpoint")
        print("✅ Checkpoint load API works")
        
    except Exception as e:
        print(f"⚠️  Demo checkpoint operations: {e}")
    
    print("\n--- Migration Requirements ---")
    
    if USE_RAY:
        print("Running with Full VERL:")
        print("  ✅ Ray initialized")
        print("  ✅ Distributed workers available")
        print("  ✅ Multi-node scaling possible")
        
    else:
        print("Running with verl-lite:")
        print("  ✅ No Ray dependencies")
        print("  ✅ Local debugging enabled")
        print("  ✅ Easy development setup")
        
    print("\n--- Code Changes for Migration ---")
    
    if USE_RAY:
        print("To switch back to verl-lite:")
        print("  1. Set USE_RAY=False")
        print("  2. No code changes needed!")
        
    else:
        print("To migrate to full VERL:")
        print("  1. Set USE_RAY=True") 
        print("  2. Ensure Ray is installed")
        print("  3. No code changes needed!")
    
    print("\n" + "="*60)
    
    cleanup()


def show_import_patterns():
    """Show the import patterns for migration."""
    
    print("\n=== IMPORT PATTERNS ===\n")
    
    print("Pattern 1: Conditional Imports")
    print("```python")
    print("if USE_RAY:")
    print("    from verl.trainer.ppo.ray_trainer import RayPPOTrainer")
    print("    trainer_cls = RayPPOTrainer")
    print("else:")
    print("    from verl_lite.trainer import LocalPPOTrainer")
    print("    trainer_cls = LocalPPOTrainer")
    print("```\n")
    
    print("Pattern 2: Direct Import (for libraries)")
    print("```python")
    print("# These imports are always the same:")
    print("from verl_lite import TensorDict, tu")
    print("from verl.trainer.config import AlgoConfig")
    print("from verl.utils.reward_score import math")
    print("```\n")
    
    print("Pattern 3: Factory Functions")
    print("```python")
    print("def create_trainer(config):")
    print("    if USE_RAY:")
    print("        return RayPPOTrainer(config)")
    print("    else:")
    print("        return LocalPPOTrainer(config)")
    print("```\n")


def main():
    """Main demonstration."""
    
    show_import_patterns()
    demonstrate_migration()
    
    print("\n🎉 Migration demonstration complete!")
    print("\nKey Takeaway:")
    print("  - Same business logic works in both environments")
    print("  - Only infrastructure changes between versions")
    print("  - Easy to develop locally, scale distributed")


if __name__ == "__main__":
    main()