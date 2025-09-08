#!/bin/bash

# Example GRPO training script for verl-lite
# This demonstrates how to run GRPO training locally without Ray

set -e

# Configuration
MODEL_PATH=${1:-"microsoft/DialoGPT-small"}  # Small model for demo
TRAIN_FILES=${2:-"example_data.parquet"}      # Replace with your data
OUTPUT_DIR=${3:-"./grpo_output"}

echo "=== GRPO Training with verl-lite ==="
echo "Model: $MODEL_PATH"
echo "Train files: $TRAIN_FILES" 
echo "Output: $OUTPUT_DIR"
echo

# Run GRPO training
python train_grpo.py \
    --model_path "$MODEL_PATH" \
    --train_files "$TRAIN_FILES" \
    --reward_function_path "reward_function.py" \
    --reward_function_name "grpo_reward_function" \
    --rollout_engine "vllm" \
    --total_epochs 2 \
    --batch_size 4 \
    --output_dir "$OUTPUT_DIR"

echo
echo "=== Training Complete ==="
echo "Checkpoints saved to: $OUTPUT_DIR/checkpoints"
echo "Logs saved to: $OUTPUT_DIR/logs"
echo
echo "To migrate to full verl:"
echo "1. Copy this recipe to verl/recipe/grpo/"
echo "2. Change imports from verl_lite to verl"
echo "3. Add Ray configuration"
echo "4. Use verl.trainer.main_ppo instead of local trainer"