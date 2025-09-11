# Migration Guide: verl-lite ↔ Full VERL

This guide explains how to migrate code between verl-lite (local development) and full VERL (distributed production).

## 🎯 Design Philosophy

verl-lite is designed around **one key principle**: 

> **Write once, migrate easily**

Code written for verl-lite should work in full VERL with minimal changes, and vice versa.

## 📋 Quick Migration Checklist

### verl-lite → Full VERL

- [ ] Change imports: `verl_lite` → `verl`
- [ ] Add Ray initialization: `ray.init()`
- [ ] Update trainer: `LocalPPOTrainer` → `RayPPOTrainer` 
- [ ] Configure distributed resources
- [ ] Update scripts to use `verl.trainer.main_ppo`

### Full VERL → verl-lite  

- [ ] Change imports: `verl` → `verl_lite`
- [ ] Remove Ray: No `ray.init()`
- [ ] Update trainer: `RayPPOTrainer` → `LocalPPOTrainer`
- [ ] Simplify to single-machine config
- [ ] Use direct Python scripts

## 🔧 Component Migration

### 1. Core Components (No Changes Needed)

These components work identically in both versions:

```python
# ✅ Same in both verl-lite and full VERL
from verl_lite import TensorDict, tu          # Data protocol
from verl.trainer.config import AlgoConfig   # Algorithm config
from verl.utils.reward_score import math     # Reward utilities
from verl.utils.tokenizer import HFTokenizer # Tokenizer
from verl.protocol import *                   # All protocol functions
```

### 2. Trainer Components

#### verl-lite Version
```python
from verl_lite.trainer import LocalPPOTrainer

config = LocalTrainingConfig(...)
trainer = LocalPPOTrainer(config)
trainer.fit(dataset)
```

#### Full VERL Version
```python
import ray
from verl.trainer.ppo.ray_trainer import RayPPOTrainer

ray.init()
config = TrainingConfig(...)  # Same config structure
trainer = RayPPOTrainer(config)
trainer.fit(dataset)
ray.shutdown()
```

### 3. Worker Components

#### verl-lite Version
```python
from verl_lite.workers import LocalFSDPWorkers

workers = LocalFSDPWorkers(config)
result = workers.generate_sequences(prompts)
```

#### Full VERL Version
```python
import ray
from verl.workers.fsdp_workers import ActorRolloutRefWorker

@ray.remote
class ActorWorker(ActorRolloutRefWorker):
    pass

actor = ActorWorker.remote(config)
result = ray.get(actor.generate_sequences.remote(prompts))
```

### 4. Rollout Components

Both versions support the same rollout engines, but use different modes:

#### verl-lite Version (Server Mode)
```python
from verl_lite.workers.rollout_local import LocalRolloutManager

# Uses HTTP server mode
rollout = LocalRolloutManager(config, engine_type="vllm")
with rollout:
    responses = rollout.generate_sequences(prompts)
```

#### Full VERL Version (Engine Mode)
```python
from verl.workers.rollout.vllm_rollout import vLLMRollout

# Uses in-process engine
rollout = vLLMRollout(config)
responses = rollout.generate_sequences(prompts)
```

## 📝 Recipe Migration

### Example: GRPO Recipe Migration

#### 1. verl-lite Recipe Structure
```
verl-lite/recipe/grpo/
├── main_grpo.py          # Training logic
├── reward_function.py    # Reward function  
├── train_grpo.py        # Script entry
└── README.md            # Documentation
```

#### 2. Migration Steps

**Step 1: Copy files**
```bash
cp -r verl-lite/recipe/grpo/ verl/recipe/grpo/
```

**Step 2: Update imports in `main_grpo.py`**
```python
# Before (verl-lite)
from verl_lite.trainer import LocalPPOTrainer
from verl_lite.workers import LocalFSDPWorkers

# After (full VERL)
from verl.trainer.ppo.ray_trainer import RayPPOTrainer
from verl.workers.fsdp_workers import ActorRolloutRefWorker, CriticWorker
```

**Step 3: Update training script**
```bash
# Before (verl-lite)
python train_grpo.py --model_path ... --train_files ...

# After (full VERL) 
python -m verl.trainer.main_ppo \
    algorithm.adv_estimator=grpo \
    data.train_files=... \
    actor_rollout_ref.model.path=...
```

**Step 4: Keep reward function identical**
```python
# ✅ No changes needed - same in both versions
def grpo_reward_function(data_source, solution_str, ground_truth, extra_info=None):
    from verl.utils.reward_score import math
    # ... exact same logic
```

## 🛠 Advanced Migration Patterns

### 1. Conditional Code Pattern

Write code that works in both environments:

```python
import os

USE_RAY = os.getenv("USE_RAY", "False").lower() == "true"

if USE_RAY:
    import ray
    from verl.trainer.ppo.ray_trainer import RayPPOTrainer
    ray.init()
    trainer_cls = RayPPOTrainer
else:
    from verl_lite.trainer import LocalPPOTrainer
    trainer_cls = LocalPPOTrainer

# Same code from here on
config = create_config()
trainer = trainer_cls(config)
trainer.fit(dataset)
```

### 2. Factory Pattern

```python
def create_trainer(config, use_ray=False):
    """Factory function for creating trainers."""
    if use_ray:
        import ray
        from verl.trainer.ppo.ray_trainer import RayPPOTrainer
        ray.init()
        return RayPPOTrainer(config)
    else:
        from verl_lite.trainer import LocalPPOTrainer
        return LocalPPOTrainer(config)
```

### 3. Configuration Abstraction

```python
class TrainingConfig:
    """Unified config that works for both versions."""
    
    def __init__(self):
        # Core settings (same for both)
        self.model_path = "microsoft/DialoGPT-small"
        self.total_epochs = 10
        self.batch_size = 32
        
        # Environment-specific settings
        if USE_RAY:
            self.n_gpus_per_node = 8
            self.nnodes = 4
        else:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
```

## 📊 Feature Compatibility Matrix

| Feature | verl-lite | Full VERL | Migration |
|---------|-----------|-----------|-----------|
| **Core Algorithm** | | | |
| PPO Training | ✅ | ✅ | ✅ None |
| GRPO Training | ✅ | ✅ | ✅ None |
| Reward Functions | ✅ | ✅ | ✅ None |
| **Data & Protocol** | | | |
| TensorDict | ✅ | ✅ | ✅ Native PyTorch |
| Datasets | ✅ | ✅ | ✅ None |
| **Models** | | | |
| FSDP Training | ✅ | ✅ | ✅ Config only |
| LoRA/QLoRA | ✅ | ✅ | ✅ None |
| Gradient Checkpointing | ✅ | ✅ | ✅ None |
| **Inference** | | | |
| vLLM | ✅ Server | ✅ Engine+Server | ⚠️ Mode change |
| SGLang | ✅ Server | ✅ Engine+Server | ⚠️ Mode change |
| HuggingFace | ✅ | ✅ | ✅ None |
| **Infrastructure** | | | |
| Single Machine | ✅ | ✅ | ✅ None |
| Multi-GPU | ✅ FSDP | ✅ FSDP+Ray | ⚠️ Ray setup |
| Multi-Node | ❌ | ✅ | 🔄 Major |
| **Development** | | | |
| Local Debugging | ✅ | ⚠️ | - |
| Easy Setup | ✅ | ⚠️ | - |
| Production Scale | ⚠️ | ✅ | - |

## 🚀 Migration Examples

### Complete Migration Example

See `examples/migration_example.py` for a full working example that demonstrates:

- Conditional imports
- Same business logic for both versions  
- Environment-specific configuration
- Unified APIs

Run it locally:
```bash
# verl-lite mode
python examples/migration_example.py

# Full VERL mode  
USE_RAY=true python examples/migration_example.py
```

### GRPO Recipe Example

See `recipe/grpo/` for a complete GRPO implementation that shows:

- Identical algorithm logic
- Same reward functions
- Compatible configuration
- Migration-ready structure

## 🐛 Troubleshooting Migration

### Common Issues

1. **Import Errors**
   ```bash
   # Error: No module named 'verl_lite'
   # Solution: Check import paths, ensure correct environment
   ```

2. **Ray Initialization**
   ```bash
   # Error: Ray not initialized
   # Solution: Add ray.init() before Ray components
   ```

3. **Configuration Mismatch**
   ```bash
   # Error: Config fields don't match
   # Solution: Use base config classes, check field names
   ```

4. **Device Issues**
   ```bash
   # Error: CUDA device mismatch
   # Solution: Ensure consistent device placement
   ```

### Migration Testing

Test your migration with these steps:

1. **Unit Tests**: Same tests should pass in both environments
2. **Integration Tests**: Run small-scale training in both modes
3. **Performance Tests**: Compare training metrics between versions
4. **Scaling Tests**: Verify distributed version scales correctly

## 📚 Best Practices

1. **Development Workflow**
   - Start with verl-lite for development (requires verl==0.5.0)
   - Use identical hyperparameters
   - Test locally before scaling
   - Migrate incrementally

2. **Code Structure**
   - Keep business logic environment-agnostic
   - Use factory patterns for infrastructure
   - Abstract configuration differences  
   - Document migration steps

3. **Testing Strategy**
   - Write tests that work in both environments
   - Use same datasets for validation
   - Compare outputs between versions
   - Automate migration testing

## 🎯 Summary

The key to successful migration is following verl-lite's design principle:

> **Infrastructure changes, algorithms don't.**

By keeping your core training logic independent of the infrastructure (Ray vs local), you can easily move between development and production environments while maintaining the same results.

**Remember**: 
- ✅ Same algorithms, same results
- ✅ Same APIs, easy migration  
- ✅ Same configs, consistent behavior
- ✅ Different infrastructure, same code