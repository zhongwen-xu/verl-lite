# Copyright 2025 Tencent
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
import subprocess
import time
import requests
from typing import List, Dict, Any, Optional
from dataclasses import dataclass

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
    
    
class LocalRolloutServer:
    """Base class for local rollout servers."""
    
    def __init__(self, config: ServerConfig, engine_type: str):
        self.config = config
        self.engine_type = engine_type
        self.server_process = None
        self.base_url = f"http://{config.host}:{config.port}"
        
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


class VLLMServerLocal(LocalRolloutServer):
    """Local vLLM server for rollout."""
    
    def __init__(self, config: ServerConfig):
        super().__init__(config, "vLLM")
    
    def start_server(self):
        """Start vLLM server."""
        logger.info(f"Starting vLLM server on {self.config.host}:{self.config.port}")
        
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


class SGLangServerLocal(LocalRolloutServer):
    """Local SGLang server for rollout."""
    
    def __init__(self, config: ServerConfig):
        super().__init__(config, "SGLang")
    
    def start_server(self):
        """Start SGLang server.""" 
        logger.info(f"Starting SGLang server on {self.config.host}:{self.config.port}")
        
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


class LocalRolloutClient(BaseRollout):
    """HTTP client for local rollout servers."""
    
    def __init__(self, config: RolloutConfig, server: LocalRolloutServer):
        self.config = config
        self.server = server
        
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


class LocalRolloutManager:
    """
    Manager for local rollout servers.
    
    Maintains the same interface as verl's rollout manager but uses
    local HTTP servers instead of Ray workers.
    """
    
    def __init__(self, config: RolloutConfig, engine_type: str = "vllm"):
        self.config = config
        self.engine_type = engine_type.lower()
        
        # Server configuration
        server_config = ServerConfig(
            model_path=getattr(config, 'model_path', ''),
            tensor_parallel_size=getattr(config, 'tensor_parallel_size', 1),
            max_model_len=getattr(config, 'max_length', None)
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
    
    def __enter__(self):
        self.start()
        return self
        
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.stop()


# Convenience function for easy usage
def create_local_rollout(config: RolloutConfig, engine_type: str = "vllm") -> LocalRolloutManager:
    """Create a local rollout manager."""
    return LocalRolloutManager(config, engine_type)