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
Local trainers for verl-lite without Ray dependencies.

These trainers maintain the same APIs as the original verl trainers
but run locally for easier debugging and prototyping.
"""

from .ppo_trainer_local import LocalPPOTrainer

__all__ = ["LocalPPOTrainer"]