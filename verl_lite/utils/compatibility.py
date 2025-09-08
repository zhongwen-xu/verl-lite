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
Compatibility utilities for mocking Ray APIs in verl-lite.

These utilities allow the same code to run in both verl-lite (local) and full verl (Ray).
"""

from typing import Any, Callable, Dict, List
import logging

logger = logging.getLogger(__name__)


class MockRayFuture:
    """Mock Ray future object that returns results immediately."""
    
    def __init__(self, result: Any):
        self._result = result
    
    def get(self):
        """Return the result immediately (no Ray overhead)."""
        return self._result
    
    def ready(self):
        """Always ready since we're running locally."""
        return True


class MockRayActor:
    """Mock Ray actor that wraps a local class instance."""
    
    def __init__(self, cls, *args, **kwargs):
        self._instance = cls(*args, **kwargs)
        self._cls_name = cls.__name__
    
    def __getattr__(self, name):
        """Delegate method calls to the wrapped instance."""
        attr = getattr(self._instance, name)
        
        if callable(attr):
            def mock_remote_method(*args, **kwargs):
                """Mock remote method call - execute locally and return MockRayFuture."""
                result = attr(*args, **kwargs)
                return MockRayFuture(result)
            return mock_remote_method
        else:
            return attr
    
    def __repr__(self):
        return f"MockRayActor({self._cls_name})"


def mock_ray_remote(cls=None, **options):
    """
    Mock ray.remote decorator for local execution.
    
    Instead of creating Ray actors, this creates local instances wrapped
    in MockRayActor to maintain the same API.
    """
    def decorator(target_cls):
        class MockRemoteClass:
            def __init__(self, *args, **kwargs):
                # Create local instance immediately
                self.actor = MockRayActor(target_cls, *args, **kwargs)
            
            def __getattr__(self, name):
                return getattr(self.actor, name)
            
            @classmethod  
            def remote(cls, *args, **kwargs):
                """Mock Ray's .remote() class method."""
                logger.debug(f"Creating local mock actor for {target_cls.__name__}")
                return cls(*args, **kwargs)
        
        MockRemoteClass.__name__ = f"Mock{target_cls.__name__}"
        MockRemoteClass.__qualname__ = f"Mock{target_cls.__qualname__}"
        
        return MockRemoteClass
    
    if cls is not None:
        # Used as @mock_ray_remote
        return decorator(cls)
    else:
        # Used as @mock_ray_remote(options...)
        return decorator


def mock_ray_get(futures):
    """
    Mock ray.get() function for local execution.
    
    Instead of waiting for Ray futures, this immediately returns results
    from MockRayFuture objects.
    """
    if isinstance(futures, list):
        results = []
        for future in futures:
            if isinstance(future, MockRayFuture):
                results.append(future.get())
            else:
                # Already a result, not a future
                results.append(future)
        return results
    else:
        # Single future
        if isinstance(futures, MockRayFuture):
            return futures.get()
        else:
            return futures


def mock_ray_wait(futures, num_returns=1, timeout=None):
    """
    Mock ray.wait() function - all futures are immediately ready.
    """
    if not isinstance(futures, list):
        futures = [futures]
    
    # In local mode, all futures are immediately ready
    ready = futures[:num_returns]
    not_ready = futures[num_returns:]
    
    return ready, not_ready


def convert_config_for_local(config):
    """
    Convert a verl config to work with verl-lite.
    
    This removes Ray-specific configurations and adjusts settings
    for local execution.
    """
    # This is a placeholder - would need to be implemented based on
    # the specific config structure of verl
    local_config = config.copy() if hasattr(config, 'copy') else config
    
    # Remove Ray-specific settings
    ray_keys_to_remove = [
        'ray_runtime_env',
        'ray_init_kwargs', 
        'resource_pool_spec',
        'colocate_actor_ref',
        'colocate_critic_actor',
    ]
    
    for key in ray_keys_to_remove:
        if hasattr(local_config, key):
            delattr(local_config, key)
    
    logger.info("Converted config for local execution")
    return local_config


# Ray API compatibility layer
class MockRayModule:
    """Mock ray module for compatibility."""
    
    @staticmethod
    def init(*args, **kwargs):
        """Mock ray.init() - no-op for local execution."""
        logger.info("Mock ray.init() called - running locally")
        return {"node_ip_address": "127.0.0.1"}
    
    @staticmethod
    def shutdown():
        """Mock ray.shutdown() - no-op for local execution."""
        logger.info("Mock ray.shutdown() called")
    
    @staticmethod
    def get_runtime_context():
        """Mock ray.get_runtime_context()."""
        class MockRuntimeContext:
            def __init__(self):
                self.worker_id = "local_worker"
                self.node_id = "local_node"
        return MockRuntimeContext()
    
    @staticmethod
    def remote(*args, **kwargs):
        return mock_ray_remote(*args, **kwargs)
    
    @staticmethod
    def get(futures):
        return mock_ray_get(futures)
    
    @staticmethod  
    def wait(futures, **kwargs):
        return mock_ray_wait(futures, **kwargs)


# Make mock ray available for imports
mock_ray = MockRayModule()


def patch_ray_imports():
    """
    Patch ray imports to use mock implementations.
    
    Call this at the beginning of your verl-lite scripts to automatically
    replace ray with mock implementations.
    """
    import sys
    sys.modules['ray'] = mock_ray
    logger.info("Ray imports patched with mock implementations")


# Convenience decorator for easy migration
def local_only(func):
    """
    Decorator to mark functions that should only run in verl-lite.
    
    This is useful for adding local-specific functionality that shouldn't
    be migrated back to full verl.
    """
    func._local_only = True
    return func


def ray_only(func):
    """
    Decorator to mark functions that should only run in full verl with Ray.
    
    These functions will raise an error in verl-lite.
    """
    def wrapper(*args, **kwargs):
        raise RuntimeError(
            f"Function {func.__name__} is only available in full verl with Ray. "
            "This function cannot be used in verl-lite."
        )
    
    wrapper._ray_only = True
    wrapper.__name__ = func.__name__
    wrapper.__doc__ = func.__doc__
    return wrapper