#!/usr/bin/env python3
"""
Minimal weight sync demo for verl-lite.

Shows how LocalPPOTrainer syncs weights to a local rollout server.
"""

import os
import logging

from verl_lite.trainer import LocalPPOTrainer
from verl_lite.trainer.ppo_trainer_local import LocalTrainingConfig

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def main():
    # Configure a tiny local run
    from dataclasses import dataclass

    @dataclass
    class MockModelCfg:
        path: str = "microsoft/DialoGPT-small"

    @dataclass
    class MockRolloutCfg:
        model_path: str = "microsoft/DialoGPT-small"
        tensor_parallel_size: int = 1
        response_length: int = 16
        temperature: float = 0.7
        top_p: float = 0.9
        do_sample: bool = True

    @dataclass
    class MockActorCfg:
        model: MockModelCfg = MockModelCfg()
        optim: type("OptimCfg", (), {"lr": 1e-5, "beta1": 0.9, "beta2": 0.95, "weight_decay": 0.0, "eps": 1e-8})()
        ppo_epochs: int = 1
        ppo_mini_batch_size: int = 2
        ppo_micro_batch_size_per_gpu: int = 1
        grad_clip: float = 1.0
        rollout: MockRolloutCfg = MockRolloutCfg()

    @dataclass
    class MockCfg(LocalTrainingConfig):
        actor_rollout_ref: MockActorCfg = MockActorCfg()
        total_epochs: int = 1
        save_freq: int = 1
        log_freq: int = 1
        sync_weights_frequency: int = 1
        enable_weight_sync: bool = True
        rollout_engine: str = "vllm"

    cfg = MockCfg()

    # Tiny dummy dataset of prompts
    from verl_lite import TensorDict
    import torch

    prompts = TensorDict({
        "input_ids": torch.randint(0, 100, (2, 8)),
        "attention_mask": torch.ones(2, 8, dtype=torch.long),
        "prompt_text": ["What is 2+2?", "Say hello"]
    }, batch_size=(2,))

    class TinyDataset:
        def __iter__(self):
            yield prompts

    trainer = LocalPPOTrainer(cfg)
    trainer.fit(TinyDataset())
    info = trainer.get_rollout_model_info()
    logger.info(f"Rollout model info after training: {info}")


if __name__ == "__main__":
    # Avoid tokenizer parallelism warning noise
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    main()

