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
verl-lite: Ray-free version of VERL for single-machine debugging and prototyping

This package provides the same APIs as the full VERL package but removes Ray dependencies
to make debugging and prototyping easier for researchers in small labs. Code written for
verl-lite can be easily migrated to full VERL with minimal changes.

Key differences from full VERL:
- No Ray dependencies - runs on single machine
- vLLM/SGLang use server mode instead of engine mode  
- Local orchestration instead of distributed workers
- Same APIs maintained for easy migration
"""

import os
import sys
import importlib

__version__ = "0.1.0"

# Import core components directly from verl
try:
    # Core protocol - import as-is since it doesn't use Ray
    from verl.protocol import DataProto
    from verl import base_config
    
    # Import logging utilities (Ray-free)
    from verl.utils import tracking
    from verl.utils.logger import aggregate_logger
    
    # Import model utilities 
    from verl.utils import model, tokenizer, torch_functional, py_functional
    
    __all__ = [
        "DataProto",
        "base_config",
        "tracking", 
        "aggregate_logger",
        "model",
        "tokenizer", 
        "torch_functional",
        "py_functional",
    ]
    
except ImportError as e:
    raise ImportError(
        f"Cannot import from verl. Make sure verl is installed: {e}\n"
        "Install with: pip install -e /path/to/verl"
    )

# Import our Ray-free components
try:
    from . import workers
    from . import trainer 
    from . import utils
    
    __all__.extend(["workers", "trainer", "utils"])
    
except ImportError as e:
    print(f"Warning: Could not import verl-lite components: {e}")

# Convenience imports that maintain verl compatibility
def __getattr__(name):
    """Provide fallback imports from verl for compatibility."""
    try:
        import verl
        if hasattr(verl, name):
            return getattr(verl, name)
    except ImportError:
        pass
    
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")