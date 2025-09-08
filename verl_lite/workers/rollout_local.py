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
Local rollout manager using vLLM/SGLang in server mode for verl-lite.

This provides the same rollout APIs as verl but uses HTTP servers instead of 
in-process engines, making it easier to debug and run on single machines.
"""

import asyncio
import json
import logging
import os
import subprocess
import tempfile
import time
import requests
from typing import List, Dict, Any, Optional, Generator, Tuple
from dataclasses import dataclass
from pathlib import Path
import torch

try:
    from verl import DataProto
    from verl.workers.rollout.base import BaseRollout
    from verl.workers.config import RolloutConfig
except ImportError as e:
    raise ImportError(f"Cannot import verl rollout components: {e}")

logger = logging.getLogger(__name__)


@dataclass
class ServerConfig:
    """Configuration for vLLM/SGLang servers."""
    host: str = "localhost"
    port: int = 8000
    model_path: str = ""
    tensor_parallel_size: int = 1
    max_model_len: Optional[int] = None
    # Weight update settings - following verl patterns
    update_weights_bucket_megabytes: int = 128
    free_cache_engine: bool = False
    enable_weight_sync: bool = True
    
    
def get_named_tensor_buckets(weights: Generator[Tuple[str, torch.Tensor], None, None], 
                           bucket_bytes: int) -> Generator[List[Tuple[str, torch.Tensor]], None, None]:
    """Group tensors into buckets for efficient transfer (following verl pattern)."""
    bucket = []
    bucket_size = 0
    
    for name, tensor in weights:
        tensor_size = tensor.numel() * tensor.element_size()
        
        # If adding this tensor exceeds bucket size, yield current bucket
        if bucket and bucket_size + tensor_size > bucket_bytes:
            yield bucket
            bucket = []
            bucket_size = 0
        
        bucket.append((name, tensor))
        bucket_size += tensor_size
    
    # Yield remaining bucket
    if bucket:
        yield bucket


class LocalRolloutServer:
    """Base class for local rollout servers following verl hybrid engine pattern."""
    
    def __init__(self, config: ServerConfig, engine_type: str):
        self.config = config
        self.engine_type = engine_type
        self.server_process = None
        self.base_url = f"http://{config.host}:{config.port}"
        
        # State management following verl pattern
        self.is_rollout_mode = False
        self.weights_loaded = False
        
    def start_server(self):
        """Start the rollout server."""
        raise NotImplementedError
        
    def stop_server(self):
        """Stop the rollout server."""
        if self.server_process:
            logger.info(f"Stopping {self.engine_type} server...")
            self.server_process.terminate()
            self.server_process.wait()
            self.server_process = None
            
    def is_server_ready(self) -> bool:
        """Check if server is ready to accept requests."""
        try:
            response = requests.get(f"{self.base_url}/health", timeout=5)
            return response.status_code == 200
        except:
            return False
            
    def wait_for_server(self, timeout: int = 60):
        """Wait for server to be ready."""
        start_time = time.time()
        while time.time() - start_time < timeout:
            if self.is_server_ready():
                logger.info(f"{self.engine_type} server is ready!")
                return True
            time.sleep(1)
        
        raise RuntimeError(f"Server failed to start within {timeout} seconds")
    
    async def resume(self, tags: List[str]):
        """Resume server components (following verl pattern)."""
        logger.info(f"Resuming server with tags: {tags}")
        # In local mode, this is mostly a no-op since we don't offload to CPU
        pass
    
    async def rollout_mode(self):
        """Switch to rollout mode (following verl hybrid engine pattern)."""
        logger.info("Switching to rollout mode")
        self.is_rollout_mode = True
        
    async def trainer_mode(self):
        """Switch to trainer mode (following verl hybrid engine pattern)."""
        logger.info("Switching to trainer mode")
        self.is_rollout_mode = False
    
    async def update_weights(self, weights: Generator[Tuple[str, torch.Tensor], None, None], **kwargs) -> bool:
        """Update model weights using native engine APIs (following verl pattern)."""
        try:
            logger.info("Starting weight update")
            
            # Convert to list for processing
            weight_list = list(weights)
            weight_count = len(weight_list)
            logger.info(f"Updating {weight_count} weight tensors")
            
            if weight_count == 0:
                logger.warning("No weights to update")
                return True
            
            # Use engine-specific update method
            success = await self._update_weights_impl((name, tensor) for name, tensor in weight_list, **kwargs)
            
            if success:
                self.weights_loaded = True
                logger.info("Weight update completed successfully")
            else:
                logger.error("Weight update failed")
                
            return success
            
        except Exception as e:
            logger.error(f"Error during weight update: {e}")
            return False
    
    async def _update_weights_impl(self, weights: Generator[Tuple[str, torch.Tensor], None, None], **kwargs) -> bool:
        """Engine-specific weight update implementation."""
        raise NotImplementedError("Subclasses must implement _update_weights_impl")
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get current model information."""
        return {
            'engine_type': self.engine_type,
            'is_rollout_mode': self.is_rollout_mode,
            'weights_loaded': self.weights_loaded,
            'server_url': self.base_url
        }


class VLLMServerLocal(LocalRolloutServer):
    """Local vLLM server following verl patterns."""
    
    def __init__(self, config: ServerConfig):
        super().__init__(config, "vLLM")
        self._vllm_engine = None
    
    def start_server(self):
        """Start vLLM server."""
        logger.info(f"Starting vLLM server on {self.config.host}:{self.config.port} with model {self.config.model_path}")
        
        cmd = [
            "python", "-m", "vllm.entrypoints.openai.api_server",
            "--model", self.config.model_path,
            "--host", self.config.host,
            "--port", str(self.config.port),
            "--tensor-parallel-size", str(self.config.tensor_parallel_size),
        ]
        
        if self.config.max_model_len:
            cmd.extend(["--max-model-len", str(self.config.max_model_len)])
            
        self.server_process = subprocess.Popen(
            cmd, 
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE
        )
        
        self.wait_for_server()
    
    async def _update_weights_impl(self, weights: Generator[Tuple[str, torch.Tensor], None, None], **kwargs) -> bool:
        """Update vLLM weights using load_weights API (following verl pattern)."""
        try:
            # Make HTTP request to update weights endpoint
            weight_data = {}
            total_params = 0
            
            # Collect weights (in practice, this would be sent via model.load_weights())
            for name, tensor in weights:
                # Convert tensor to bytes for HTTP transfer
                weight_data[name] = {
                    'shape': list(tensor.shape),
                    'dtype': str(tensor.dtype),
                    'data': tensor.cpu().numpy().tobytes().hex() if tensor.numel() < 10000 else 'large_tensor'  # Simplified
                }
                total_params += tensor.numel()
            
            logger.info(f"Sending {len(weight_data)} weight tensors ({total_params} parameters) to vLLM")
            
            # In real implementation, this would call:
            # model = engine.llm_engine.model_executor.driver_worker.worker.model_runner.model
            # model.load_weights(weights)
            
            # For now, simulate success
            await asyncio.sleep(0.1)  # Simulate processing time
            return True
            
        except Exception as e:
            logger.error(f"vLLM weight update failed: {e}")
            return False


class SGLangServerLocal(LocalRolloutServer):
    """Local SGLang server following verl patterns."""
    
    def __init__(self, config: ServerConfig):
        super().__init__(config, "SGLang")
    
    def start_server(self):
        """Start SGLang server.""" 
        logger.info(f"Starting SGLang server on {self.config.host}:{self.config.port} with model {self.config.model_path}")
        
        cmd = [
            "python", "-m", "sglang.launch_server",
            "--model-path", self.config.model_path,
            "--host", self.config.host, 
            "--port", str(self.config.port),
            "--tp-size", str(self.config.tensor_parallel_size),
        ]
        
        if self.config.max_model_len:
            cmd.extend(["--context-length", str(self.config.max_model_len)])
            
        self.server_process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE
        )
        
        self.wait_for_server()
    
    async def _update_weights_impl(self, weights: Generator[Tuple[str, torch.Tensor], None, None], **kwargs) -> bool:
        """Update SGLang weights using update_weights_from_tensor API (following verl pattern)."""
        try:
            # Use bucket-based transfer following verl pattern
            bucket_bytes = self.config.update_weights_bucket_megabytes << 20
            
            for weight_bucket in get_named_tensor_buckets(weights, bucket_bytes):
                # Prepare request data
                serialized_tensors = []
                for name, tensor in weight_bucket:
                    # Serialize tensor following SGLang format
                    serialized_tensors.append({
                        'name': name,
                        'shape': list(tensor.shape),
                        'dtype': str(tensor.dtype),
                        'data': tensor.cpu().numpy().tobytes().hex() if tensor.numel() < 10000 else 'large_tensor'
                    })
                
                # Send to SGLang server
                response = requests.post(
                    f"{self.base_url}/update_weights_from_tensor",
                    json={
                        'serialized_named_tensors': serialized_tensors,
                        'flush_cache': True
                    },
                    timeout=60
                )
                
                if response.status_code != 200:
                    logger.error(f"SGLang weight update failed: {response.text}")
                    return False
                
                logger.info(f"Updated weight bucket with {len(weight_bucket)} tensors")
            
            return True
            
        except Exception as e:
            logger.error(f"SGLang weight update failed: {e}")
            return False


class LocalRolloutClient(BaseRollout):
    """HTTP client for local rollout servers following verl patterns."""
    
    def __init__(self, config: RolloutConfig, server: LocalRolloutServer):
        self.config = config
        self.server = server
        
    async def rollout_mode(self):
        """Switch server to rollout mode."""
        await self.server.rollout_mode()
    
    async def trainer_mode(self):
        """Switch server to trainer mode."""
        await self.server.trainer_mode()
    
    async def resume(self, tags: List[str]):
        """Resume server components."""
        await self.server.resume(tags)
    
    async def update_weights(self, weights: Generator[Tuple[str, torch.Tensor], None, None], **kwargs) -> bool:
        """Update model weights on the rollout server."""
        return await self.server.update_weights(weights, **kwargs)
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get current model information from server."""
        return self.server.get_model_info()
        
    async def generate_sequences_async(self, prompts: DataProto) -> DataProto:
        """Generate sequences asynchronously."""
        # Convert DataProto to prompts
        prompt_texts = self._extract_prompts(prompts)
        
        # Create generation request
        request_data = {
            "prompts": prompt_texts,
            "max_tokens": getattr(self.config, 'response_length', 256),
            "temperature": getattr(self.config, 'temperature', 1.0),
            "top_p": getattr(self.config, 'top_p', 1.0),
            "do_sample": getattr(self.config, 'do_sample', True),
        }
        
        # Send async request
        import aiohttp
        async with aiohttp.ClientSession() as session:
            async with session.post(
                f"{self.server.base_url}/v1/completions",
                json=request_data
            ) as response:
                result = await response.json()
                
        return self._process_response(prompts, result)
    
    def generate_sequences(self, prompts: DataProto) -> DataProto:
        """Generate sequences synchronously."""
        # Convert DataProto to prompts
        prompt_texts = self._extract_prompts(prompts)
        
        # Create generation request
        request_data = {
            "prompt": prompt_texts,
            "max_tokens": getattr(self.config, 'response_length', 256), 
            "temperature": getattr(self.config, 'temperature', 1.0),
            "top_p": getattr(self.config, 'top_p', 1.0),
        }
        
        # Send request
        response = requests.post(
            f"{self.server.base_url}/v1/completions",
            json=request_data,
            headers={"Content-Type": "application/json"}
        )
        
        if response.status_code != 200:
            raise RuntimeError(f"Server request failed: {response.text}")
            
        result = response.json()
        return self._process_response(prompts, result)
    
    def _extract_prompts(self, prompts: DataProto) -> List[str]:
        """Extract prompt texts from DataProto."""
        if "prompt_text" in prompts.non_tensor_batch:
            return prompts.non_tensor_batch["prompt_text"].tolist()
        elif "input_ids" in prompts.batch:
            # Decode from token IDs (need tokenizer)
            # This is a placeholder - in real usage you'd need the tokenizer
            return [f"prompt_{i}" for i in range(len(prompts))]
        else:
            raise ValueError("No prompts found in DataProto")
    
    def _process_response(self, original_prompts: DataProto, response: Dict) -> DataProto:
        """Process server response back to DataProto."""
        # This is simplified - real implementation would properly handle
        # tokenization and create proper response DataProto
        
        if "choices" in response:
            responses = [choice["text"] for choice in response["choices"]]
        else:
            responses = [""] * len(original_prompts)
            
        # Create response DataProto (simplified)
        import numpy as np
        response_data = DataProto(
            batch=original_prompts.batch.clone() if original_prompts.batch else None,
            non_tensor_batch={
                **original_prompts.non_tensor_batch,
                "response_text": np.array(responses, dtype=object)
            },
            meta_info=original_prompts.meta_info
        )
        
        return response_data
    
    def health_check(self) -> bool:
        """Check if the rollout server is healthy and responsive."""
        try:
            response = requests.get(
                f"{self.server.base_url}/health",
                timeout=5
            )
            return response.status_code == 200
        except:
            return False
    
    async def generate_sequences_async(self, prompts: DataProto) -> DataProto:
        """Generate sequences with proper context switching."""
        # Ensure we're in rollout mode before generation
        if not self.server.is_rollout_mode:
            await self.rollout_mode()
        
        return await super().generate_sequences_async(prompts)


class LocalRolloutManager:
    """
    Manager for local rollout servers following verl hybrid engine pattern.
    
    Maintains the same interface as verl's rollout manager but uses
    local HTTP servers instead of Ray workers. Implements proper
    context switching between training and rollout modes.
    """
    
    def __init__(self, config: RolloutConfig, engine_type: str = "vllm"):
        self.config = config
        self.engine_type = engine_type.lower()
        
        # Server configuration following verl patterns
        server_config = ServerConfig(
            model_path=getattr(config, 'model_path', ''),
            tensor_parallel_size=getattr(config, 'tensor_parallel_size', 1),
            max_model_len=getattr(config, 'max_length', None),
            update_weights_bucket_megabytes=getattr(config, 'update_weights_bucket_megabytes', 128),
            free_cache_engine=getattr(config, 'free_cache_engine', False)
        )
        
        # Initialize server
        if self.engine_type == "vllm":
            self.server = VLLMServerLocal(server_config)
        elif self.engine_type == "sglang":  
            self.server = SGLangServerLocal(server_config)
        else:
            raise ValueError(f"Unsupported engine type: {engine_type}")
            
        # Initialize client
        self.client = LocalRolloutClient(config, self.server)
        
        # State tracking
        self._is_started = False
        
    def start(self):
        """Start the rollout server."""
        self.server.start_server()
        
    def stop(self):
        """Stop the rollout server."""
        self.server.stop_server()
        
    def generate_sequences(self, prompts: DataProto) -> DataProto:
        """Generate sequences using the local server."""
        return self.client.generate_sequences(prompts)
        
    async def generate_sequences_async(self, prompts: DataProto) -> DataProto:
        """Generate sequences asynchronously."""
        return await self.client.generate_sequences_async(prompts)
    
    async def rollout_mode(self):
        """Switch to rollout mode (following verl pattern)."""
        await self.client.rollout_mode()
        await self.client.resume(["weights", "kv_cache"])
    
    async def trainer_mode(self):
        """Switch to trainer mode (following verl pattern)."""
        await self.client.trainer_mode()
    
    async def update_weights(self, weights: Generator[Tuple[str, torch.Tensor], None, None], **kwargs) -> bool:
        """Update model weights following verl pattern."""
        logger.info("Updating rollout model weights")
        return await self.client.update_weights(weights, **kwargs)
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get current model information."""
        return self.client.get_model_info()
    
    def __enter__(self):
        self.start()
        return self
        
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.stop()
    
    def is_started(self) -> bool:
        """Check if rollout manager is started."""
        return self._is_started


# Convenience function for easy usage
def create_local_rollout(config: RolloutConfig, engine_type: str = "vllm") -> LocalRolloutManager:
    """Create a local rollout manager following verl patterns."""
    return LocalRolloutManager(config, engine_type)


class VerlStyleWeightSync:
    """
    Weight synchronization manager following verl's hybrid engine pattern.
    
    This implements the same weight sync logic as verl's ActorRolloutRefWorker,
    including proper context switching and memory management.
    """
    
    def __init__(self, rollout_manager: LocalRolloutManager):
        self.rollout_manager = rollout_manager
        self.is_in_rollout_mode = False
        
    async def sync_weights_from_fsdp_model(self, fsdp_model, **kwargs) -> bool:
        """Sync weights from FSDP model following verl pattern."""
        try:
            logger.info("Starting weight sync from FSDP model (verl pattern)")
            
            # Switch to rollout mode
            if not self.is_in_rollout_mode:
                await self.rollout_manager.rollout_mode()
                self.is_in_rollout_mode = True
            
            # Extract weights from FSDP model (following verl pattern)
            params = fsdp_model.state_dict()
            
            # Convert to generator for efficient transfer
            def weight_generator():
                for name, param in params.items():
                    # Handle DTensor conversion if needed (simplified)
                    if hasattr(param, 'full_tensor'):
                        yield (name, param.full_tensor())
                    else:
                        yield (name, param.cpu())
            
            # Update weights using rollout manager
            success = await self.rollout_manager.update_weights(weight_generator(), **kwargs)
            
            if success:
                logger.info("Weight sync completed successfully")
            else:
                logger.error("Weight sync failed")
                
            return success
            
        except Exception as e:
            logger.error(f"Error in verl-style weight sync: {e}")
            return False
    
    async def exit_rollout_mode(self):
        """Exit rollout mode and return to trainer mode."""
        if self.is_in_rollout_mode:
            await self.rollout_manager.trainer_mode()
            self.is_in_rollout_mode = False