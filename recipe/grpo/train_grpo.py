#!/usr/bin/env python3
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
Simple GRPO training script for verl-lite.

This demonstrates the same GRPO training as the full verl recipe but runs locally.
"""

import os
import sys

# Add current directory to path for importing local modules
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from main_grpo import main

if __name__ == "__main__":
    # Set some default environment variables for local development
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    
    # Run the main GRPO training
    main()