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
Reward function for GRPO training.

This matches the same reward function used in verl for easy migration.
"""


def grpo_reward_function(data_source, solution_str, ground_truth, extra_info=None):
    """
    Math accuracy reward function for GRPO.
    
    This uses the same logic as verl's char_count reward function.
    """
    try:
        # Import verl's math utilities (same as in full verl)
        from verl.utils.reward_score import math
        
        # Extract the last boxed answer from the solution
        last_boxed_string = math.last_boxed_only_string(solution_str)
        if last_boxed_string is None:
            return 0
        
        solution = math.remove_boxed(last_boxed_string)
        
        # Check if solution matches ground truth exactly
        if solution == ground_truth:
            return 1
        else:
            return 0
            
    except Exception:
        # Fallback in case of any parsing errors
        print(f"Reward parsing failed - ground_truth: {ground_truth}, solution: {solution_str}")
        return 0