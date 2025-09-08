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
Validation of verl-lite implementation against original verl patterns.

This script validates that verl-lite correctly implements the same 
weight synchronization patterns as the original verl codebase.
"""

import asyncio
import logging
import torch
import torch.nn as nn
from typing import Generator, Tuple

# Import verl-lite components
from verl_lite.workers.rollout_local import (
    LocalRolloutManager, 
    VerlStyleWeightSync,
    get_named_tensor_buckets
)

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class MockFSDPModel(nn.Module):
    """Mock FSDP model for testing."""
    
    def __init__(self):
        super().__init__()
        self.linear1 = nn.Linear(100, 50)
        self.linear2 = nn.Linear(50, 10)
    
    def forward(self, x):
        x = self.linear1(x)
        return self.linear2(x)


def create_mock_rollout_config():
    """Create a mock rollout config similar to verl."""
    from dataclasses import dataclass
    
    @dataclass
    class MockRolloutConfig:
        model_path: str = "gpt2"
        tensor_parallel_size: int = 1
        max_length: int = 512
        update_weights_bucket_megabytes: int = 64
        free_cache_engine: bool = False
    
    return MockRolloutConfig()


async def test_weight_bucket_functionality():
    """Test the weight bucketing functionality following verl patterns."""
    logger.info("Testing weight bucketing functionality")
    
    # Create mock weights
    def create_mock_weights() -> Generator[Tuple[str, torch.Tensor], None, None]:
        for i in range(10):
            # Create tensors of different sizes
            size = 1000 * (i + 1)
            tensor = torch.randn(size)
            yield (f"layer_{i}.weight", tensor)
    
    # Test bucketing
    bucket_bytes = 50000  # 50KB buckets
    buckets = list(get_named_tensor_buckets(create_mock_weights(), bucket_bytes))
    
    logger.info(f"Created {len(buckets)} buckets")
    for i, bucket in enumerate(buckets):
        total_size = sum(tensor.numel() * tensor.element_size() for _, tensor in bucket)
        logger.info(f"Bucket {i}: {len(bucket)} tensors, {total_size} bytes")
    
    return len(buckets) > 0


async def test_verl_style_weight_sync():
    """Test verl-style weight synchronization."""
    logger.info("Testing verl-style weight synchronization")
    
    try:
        # Create rollout manager
        config = create_mock_rollout_config()
        rollout_manager = LocalRolloutManager(config, engine_type="vllm")
        
        # Create weight sync manager
        weight_sync = VerlStyleWeightSync(rollout_manager)
        
        # Create mock FSDP model
        mock_model = MockFSDPModel()
        
        # Test weight sync
        success = await weight_sync.sync_weights_from_fsdp_model(mock_model)
        
        if success:
            logger.info("✓ Verl-style weight sync successful")
        else:
            logger.warning("✗ Verl-style weight sync failed")
        
        # Test mode switching
        await weight_sync.exit_rollout_mode()
        logger.info("✓ Successfully exited rollout mode")
        
        return success
        
    except Exception as e:
        logger.error(f"Error in verl-style weight sync test: {e}")
        return False


async def test_hybrid_engine_context_switching():
    """Test hybrid engine context switching following verl patterns."""
    logger.info("Testing hybrid engine context switching")
    
    try:
        # Create rollout manager
        config = create_mock_rollout_config()
        rollout_manager = LocalRolloutManager(config, engine_type="sglang")
        
        # Test context switching
        await rollout_manager.rollout_mode()
        model_info = rollout_manager.get_model_info()
        
        if model_info.get('is_rollout_mode', False):
            logger.info("✓ Successfully switched to rollout mode")
        else:
            logger.warning("✗ Failed to switch to rollout mode")
            return False
        
        await rollout_manager.trainer_mode()
        model_info = rollout_manager.get_model_info()
        
        if not model_info.get('is_rollout_mode', True):
            logger.info("✓ Successfully switched to trainer mode")
        else:
            logger.warning("✗ Failed to switch to trainer mode")
            return False
        
        return True
        
    except Exception as e:
        logger.error(f"Error in context switching test: {e}")
        return False


async def test_weight_update_apis():
    """Test weight update APIs for both vLLM and SGLang."""
    logger.info("Testing weight update APIs")
    
    results = {}
    
    for engine_type in ["vllm", "sglang"]:
        try:
            logger.info(f"Testing {engine_type} weight updates")
            
            # Create rollout manager
            config = create_mock_rollout_config()
            rollout_manager = LocalRolloutManager(config, engine_type=engine_type)
            
            # Create mock weights
            def create_weights():
                model = MockFSDPModel()
                for name, param in model.state_dict().items():
                    yield (name, param)
            
            # Test weight update
            success = await rollout_manager.update_weights(create_weights())
            results[engine_type] = success
            
            if success:
                logger.info(f"✓ {engine_type} weight update successful")
            else:
                logger.warning(f"✗ {engine_type} weight update failed")
                
        except Exception as e:
            logger.error(f"Error testing {engine_type} weight updates: {e}")
            results[engine_type] = False
    
    return all(results.values())


async def test_memory_management_patterns():
    """Test memory management patterns following verl."""
    logger.info("Testing memory management patterns")
    
    try:
        # Create rollout manager
        config = create_mock_rollout_config()
        config.free_cache_engine = True  # Enable memory management
        
        rollout_manager = LocalRolloutManager(config, engine_type="vllm")
        
        # Test resume functionality
        await rollout_manager.client.resume(["weights"])
        logger.info("✓ Resumed weights successfully")
        
        await rollout_manager.client.resume(["kv_cache"])
        logger.info("✓ Resumed kv_cache successfully")
        
        return True
        
    except Exception as e:
        logger.error(f"Error in memory management test: {e}")
        return False


async def validate_verl_patterns():
    """Main validation function."""
    logger.info("=" * 60)
    logger.info("verl-lite Pattern Validation")
    logger.info("=" * 60)
    
    tests = [
        ("Weight Bucketing", test_weight_bucket_functionality),
        ("Verl-Style Weight Sync", test_verl_style_weight_sync),
        ("Hybrid Engine Context Switching", test_hybrid_engine_context_switching),
        ("Weight Update APIs", test_weight_update_apis),
        ("Memory Management", test_memory_management_patterns),
    ]
    
    results = {}
    
    for test_name, test_func in tests:
        logger.info(f"\n--- {test_name} ---")
        try:
            success = await test_func()
            results[test_name] = success
            
            if success:
                logger.info(f"✅ {test_name}: PASSED")
            else:
                logger.error(f"❌ {test_name}: FAILED")
                
        except Exception as e:
            logger.error(f"❌ {test_name}: ERROR - {e}")
            results[test_name] = False
    
    # Summary
    logger.info("\n" + "=" * 60)
    logger.info("VALIDATION SUMMARY")
    logger.info("=" * 60)
    
    passed = sum(1 for success in results.values() if success)
    total = len(results)
    
    for test_name, success in results.items():
        status = "✅ PASSED" if success else "❌ FAILED"
        logger.info(f"{test_name}: {status}")
    
    logger.info(f"\nOverall: {passed}/{total} tests passed")
    
    if passed == total:
        logger.info("🎉 All verl pattern validations PASSED!")
        return True
    else:
        logger.warning("⚠️  Some validations FAILED - review implementation")
        return False


def main():
    """Run validation tests."""
    try:
        # Run async validation
        success = asyncio.run(validate_verl_patterns())
        
        if success:
            print("\n🎉 verl-lite successfully implements verl patterns!")
            exit(0)
        else:
            print("\n❌ verl-lite implementation needs fixes")
            exit(1)
            
    except Exception as e:
        logger.error(f"Validation failed with error: {e}")
        exit(1)


if __name__ == "__main__":
    main()