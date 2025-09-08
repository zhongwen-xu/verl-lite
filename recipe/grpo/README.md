# GRPO Recipe for verl-lite

This example demonstrates how to implement GRPO (Group Relative Policy Optimization) using verl-lite components.

## Key Features

- **Same API as full verl**: Uses identical interfaces and components
- **Local execution**: Runs without Ray for easier debugging
- **Easy migration**: Can be copied to full verl with minimal changes
- **Server-mode inference**: Uses vLLM/SGLang in HTTP server mode

## Files

- `main_grpo.py`: Main GRPO training logic
- `reward_function.py`: Reward function (same as verl)
- `train_grpo.py`: Training script entry point
- `run_grpo_example.sh`: Example shell script
- `README.md`: This documentation

## Usage

### Basic Usage

```bash
# Run with default small model (for testing)
./run_grpo_example.sh

# Run with your own model and data
./run_grpo_example.sh "path/to/your/model" "path/to/train.parquet" "./output"
```

### Python Usage

```bash
python train_grpo.py \
    --model_path "microsoft/DialoGPT-small" \
    --train_files "train_data.parquet" \
    --reward_function_path "reward_function.py" \
    --reward_function_name "grpo_reward_function" \
    --rollout_engine "vllm" \
    --total_epochs 2 \
    --batch_size 4 \
    --output_dir "./grpo_output"
```

## Configuration

The GRPO trainer supports the same configurations as full verl:

- **Algorithm settings**: gamma, lambda, epsilon, entropy coefficient
- **Model settings**: path, flash attention, gradient checkpointing  
- **Data settings**: batch size, sequence lengths, file paths
- **Rollout settings**: engine type (vLLM/SGLang), generation parameters

## Data Format

Training data should be in parquet format with columns:
- `prompt`: Input prompts for the model
- `ground_truth`: Expected answers for reward computation

Example:
```python
import pandas as pd

data = pd.DataFrame({
    'prompt': ['What is 2+2?', 'Solve: 3*4'],
    'ground_truth': ['4', '12']
})
data.to_parquet('train_data.parquet')
```

## Migration to Full VERL

To migrate this recipe to full verl for distributed training:

### 1. Copy Files
```bash
cp -r verl-lite/recipe/grpo/ verl/recipe/grpo/
```

### 2. Update Imports
```python
# Change from:
from verl_lite.trainer import LocalPPOTrainer
from verl_lite.workers import LocalFSDPWorkers

# To:
from verl.trainer.ppo.ray_trainer import RayPPOTrainer
from verl.workers.fsdp_workers import ActorRolloutRefWorker, CriticWorker
```

### 3. Use Ray Trainer
```python
# Change from:
trainer = LocalPPOTrainer(config)

# To:
import ray
ray.init()
trainer = RayPPOTrainer(config)
```

### 4. Update Script
```bash
# Change from local script to verl's main entry:
python -m verl.trainer.main_ppo \
    algorithm.adv_estimator=grpo \
    # ... other hydra configs
```

The algorithm logic and reward function remain exactly the same!

## Architecture

```
GRPO Recipe Structure:

┌─ main_grpo.py           # Main training logic
├─ reward_function.py     # Reward computation  
├─ train_grpo.py         # Script entry point
└─ GRPOTrainer
   ├─ LocalPPOTrainer    # Local orchestration
   ├─ LocalFSDPWorkers   # FSDP model workers
   └─ LocalRolloutManager # vLLM/SGLang server
```

## Comparison with Full VERL

| Feature | verl-lite | Full VERL |
|---------|-----------|-----------|
| Algorithm | ✅ Same GRPO | ✅ Same GRPO |
| Reward Function | ✅ Same | ✅ Same |
| Model Training | ✅ FSDP | ✅ FSDP + Ray |
| Rollout Engine | ✅ Server mode | ✅ Engine + Server |
| Distributed | ❌ Single machine | ✅ Multi-node |
| Debugging | ✅ Easy | ⚠️ Complex |

## Tips

1. **Start Small**: Use small models and datasets for initial development
2. **Debug Locally**: Develop and debug with verl-lite first
3. **Scale Up**: Migrate to full verl when ready for production
4. **Same Configs**: Use the same hyperparameters in both versions