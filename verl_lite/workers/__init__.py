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
Ray-free workers for verl-lite.

These workers provide the same APIs as the original verl workers but without Ray dependencies.
They run locally on a single machine for debugging and prototyping.
"""

# Import non-Ray components directly from verl
try:
    from verl.workers.config import *
    from verl.workers.roles import *
except ImportError:
    pass

# Import our Ray-free implementations
from .fsdp_workers_local import LocalFSDPWorkers
from .rollout_local import LocalRolloutManager

__all__ = [
    "LocalFSDPWorkers",
    "LocalRolloutManager",
]