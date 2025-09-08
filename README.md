# verl-lite

Ray-free version of [VERL](https://github.com/volcengine/verl) for single-machine debugging and prototyping.

## 🎯 Purpose

verl-lite enables researchers in small labs to:
- **Debug locally** without Ray complexity
- **Prototype quickly** with same APIs as full VERL  
- **Migrate easily** to distributed VERL for scaling
- **Use optimized inference** with vLLM/SGLang in server mode

## 🏗️ Design Principle

> **Write once, migrate easily**

Code written for verl-lite works in full VERL with minimal changes.

## 🚀 Quick Start

### Installation

```bash
# Basic installation (requires verl==0.5.0)
pip install -e .

# With vLLM support
pip install -e .[vllm]

# With SGLang support  
pip install -e .[sglang]

# Everything
pip install -e .[all]
```

### Basic Usage

```python
import verl_lite

# Import verl components directly (same API)
from verl import DataProto
from verl.trainer.config import AlgoConfig

# Use verl-lite local components
from verl_lite.trainer import LocalPPOTrainer
from verl_lite.workers import LocalFSDPWorkers
```

### GRPO Example

```bash
# Run GRPO training locally
cd recipe/grpo
./run_grpo_example.sh "microsoft/DialoGPT-small" "train.parquet" "./output"
```

### Weight Synchronization

```python
# Weight updates are automatically synced from trainer to rollout server
from verl_lite.trainer import LocalPPOTrainer, LocalTrainingConfig

config = LocalTrainingConfig(
    enable_weight_sync=True,
    sync_weights_frequency=1  # Sync every training step
)

trainer = LocalPPOTrainer(config)
# Weights automatically sync during training steps
```

## 🔧 Architecture

verl-lite maintains the same APIs but removes Ray:

```
┌─ verl Components (Direct Import)
│  ├─ DataProto           # Same data protocol
│  ├─ AlgoConfig          # Same algorithm config
│  └─ reward_score        # Same reward utilities
│
├─ verl-lite Components (Local Execution)  
│  ├─ LocalPPOTrainer     # PPO without Ray
│  ├─ LocalFSDPWorkers    # FSDP without Ray
│  └─ LocalRolloutManager # Server-mode inference
│
└─ Migration Tools
   ├─ Compatibility utils  # Mock Ray APIs
   └─ Migration helpers   # Auto-convert code
```

## 📊 Comparison with Full VERL

| Feature | verl-lite | Full VERL | Migration |
|---------|-----------|-----------|-----------|
| **Algorithms** | ✅ Same | ✅ Same | ✅ None |
| **Data Protocol** | ✅ Same | ✅ Same | ✅ None |
| **Weight Sync** | ✅ Auto-sync | ✅ Ray-based | ⚠️ Config |
| **FSDP Training** | ✅ Single machine | ✅ Distributed | ⚠️ Config |
| **vLLM/SGLang** | ✅ Server mode | ✅ Engine+Server | ⚠️ Mode |
| **Ray** | ❌ None | ✅ Required | 🔄 Add Ray |
| **Debugging** | ✅ Easy | ⚠️ Complex | - |
| **Scaling** | ⚠️ Limited | ✅ Multi-node | - |

## 🧪 Examples

### 1. GRPO Recipe
Complete GRPO implementation that can be migrated to full VERL:
- `recipe/grpo/main_grpo.py` - Training logic
- `recipe/grpo/reward_function.py` - Reward function
- `recipe/grpo/README.md` - Documentation

### 2. Weight Synchronization Demo
Demonstrates automatic weight updates during RL training:
- `examples/weight_sync_example.py` - Weight sync demonstration
- Automatic sync from trainer to rollout server
- Critical for RL training effectiveness

### 3. Migration Example  
Demonstrates same code working in both environments:
- `examples/migration_example.py` - Conditional execution

## 🔄 Migration to Full VERL

### Quick Migration
1. **Change imports**:
   ```python
   # From:
   from verl_lite.trainer import LocalPPOTrainer
   # To:
   from verl.trainer.ppo.ray_trainer import RayPPOTrainer
   ```

2. **Add Ray**:
   ```python
   import ray
   ray.init()
   ```

3. **Same everything else** - configs, algorithms, rewards all identical!

### Recipe Migration
```bash
# Copy recipe to full verl
cp -r verl-lite/recipe/grpo/ verl/recipe/grpo/

# Update imports and run with full verl
python -m verl.trainer.main_ppo algorithm.adv_estimator=grpo ...
```

See `MIGRATION_GUIDE.md` for detailed instructions.

## 📁 Project Structure

```
verl-lite/
├── verl_lite/
│   ├── workers/          # Ray-free workers
│   ├── trainer/          # Local trainers  
│   └── utils/            # Migration utilities
├── recipe/
│   └── grpo/             # GRPO example
├── examples/             # Usage examples
├── MIGRATION_GUIDE.md    # Migration instructions
└── README.md             # This file
```

## 🎯 Use Cases

### Perfect for:
- 🔬 **Algorithm research** - Focus on RL without infrastructure
- 🐛 **Local debugging** - Easy breakpoints and inspection
- 🚀 **Rapid prototyping** - Fast iteration cycles
- 📚 **Learning VERL** - Understand concepts without complexity

### Not ideal for:
- 🏭 **Production training** - Use full VERL instead
- 🌐 **Multi-node scaling** - Requires Ray infrastructure
- 📊 **Large-scale experiments** - Limited by single machine

## 🤝 Contributing

Since this is part of the VERL ecosystem:
1. Keep APIs identical to full VERL
2. Ensure easy migration path
3. Test with both local and distributed versions
4. Document migration steps

## 📄 License

Apache 2.0 (same as VERL)

## 🙏 Acknowledgments

Built on top of the amazing [VERL](https://github.com/volcengine/verl) project by the Bytedance team.

---

**Remember**: Start with verl-lite for development, migrate to full VERL for production! 🚀