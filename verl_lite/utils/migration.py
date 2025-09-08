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
Migration utilities for moving code between verl-lite and full verl.
"""

import ast
import os
from typing import List, Dict, Any


class CodeMigrationAnalyzer:
    """Analyze code for migration compatibility between verl-lite and full verl."""
    
    def __init__(self):
        self.verl_lite_imports = {
            'verl_lite',
            'verl_lite.workers',
            'verl_lite.trainer', 
            'verl_lite.utils',
        }
        
        self.ray_patterns = {
            'ray.remote',
            'ray.get',
            'ray.wait',
            'ray.init',
            'ray.shutdown',
            '@ray.remote',
        }
        
        self.local_only_patterns = {
            'LocalPPOTrainer',
            'LocalFSDPWorkers',
            'LocalRolloutManager',
            'mock_ray_',
        }
    
    def analyze_file(self, file_path: str) -> Dict[str, Any]:
        """Analyze a Python file for migration compatibility."""
        with open(file_path, 'r') as f:
            content = f.read()
        
        tree = ast.parse(content)
        
        analysis = {
            'file': file_path,
            'verl_lite_imports': [],
            'ray_usage': [],
            'local_only_features': [],
            'migration_complexity': 'easy',
            'required_changes': [],
        }
        
        # Check imports and usage
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    if any(pattern in alias.name for pattern in self.verl_lite_imports):
                        analysis['verl_lite_imports'].append(alias.name)
                    
            elif isinstance(node, ast.ImportFrom):
                if node.module and any(pattern in node.module for pattern in self.verl_lite_imports):
                    analysis['verl_lite_imports'].append(node.module)
                    
            elif isinstance(node, ast.Call):
                if isinstance(node.func, ast.Attribute):
                    func_name = f"{ast.unparse(node.func.value)}.{node.func.attr}"
                    if any(pattern in func_name for pattern in self.ray_patterns):
                        analysis['ray_usage'].append(func_name)
                        
            elif isinstance(node, ast.Name):
                if any(pattern in node.id for pattern in self.local_only_patterns):
                    analysis['local_only_features'].append(node.id)
        
        # Determine migration complexity
        if analysis['local_only_features']:
            analysis['migration_complexity'] = 'hard'
            analysis['required_changes'].append("Replace local-only components with Ray versions")
        elif analysis['verl_lite_imports']:
            analysis['migration_complexity'] = 'medium'  
            analysis['required_changes'].append("Change imports from verl_lite to verl")
        else:
            analysis['migration_complexity'] = 'easy'
            analysis['required_changes'].append("No major changes needed")
            
        return analysis
    
    def analyze_directory(self, dir_path: str) -> List[Dict[str, Any]]:
        """Analyze all Python files in a directory."""
        results = []
        
        for root, dirs, files in os.walk(dir_path):
            for file in files:
                if file.endswith('.py'):
                    file_path = os.path.join(root, file)
                    try:
                        analysis = self.analyze_file(file_path)
                        results.append(analysis)
                    except Exception as e:
                        results.append({
                            'file': file_path,
                            'error': str(e),
                            'migration_complexity': 'error'
                        })
        
        return results


def create_migration_guide(source_dir: str, output_file: str = "MIGRATION_GUIDE.md"):
    """Create a migration guide for moving code from verl-lite to full verl."""
    
    analyzer = CodeMigrationAnalyzer()
    analyses = analyzer.analyze_directory(source_dir)
    
    guide_content = f"""# Migration Guide: verl-lite to Full VERL

This guide helps you migrate code from verl-lite back to the full VERL package for production deployment.

## Overview

verl-lite is designed to make migration back to full VERL as easy as possible. Most code should work with minimal changes.

## Migration Steps

### 1. Update Imports

Change all imports from `verl_lite` to `verl`:

```python
# verl-lite version
from verl_lite.trainer import LocalPPOTrainer
from verl_lite.workers import LocalFSDPWorkers

# Full verl version  
from verl.trainer.ppo.ray_trainer import RayPPOTrainer
from verl.workers.fsdp_workers import ActorRolloutRefWorker, CriticWorker
```

### 2. Update Worker Usage

Replace local workers with Ray workers:

```python
# verl-lite version
workers = LocalFSDPWorkers(config)

# Full verl version
import ray
@ray.remote
class ActorWorker:
    # ... Ray worker implementation
```

### 3. Update Trainer

Replace local trainer with Ray trainer:

```python
# verl-lite version
trainer = LocalPPOTrainer(config)

# Full verl version  
trainer = RayPPOTrainer(config)
```

### 4. Configure Ray

Add Ray configuration to your training script:

```python
import ray

# Initialize Ray
ray.init()

# Your training code here

# Shutdown Ray
ray.shutdown()
```

## File Analysis

"""
    
    # Add file-by-file analysis
    for analysis in analyses:
        if 'error' in analysis:
            guide_content += f"\n### {analysis['file']} (ERROR)\n"
            guide_content += f"Error analyzing file: {analysis['error']}\n"
            continue
            
        complexity = analysis['migration_complexity']
        guide_content += f"\n### {analysis['file']} ({complexity.upper()})\n"
        
        if analysis['verl_lite_imports']:
            guide_content += f"\n**verl-lite imports to change:**\n"
            for imp in analysis['verl_lite_imports']:
                guide_content += f"- `{imp}`\n"
                
        if analysis['local_only_features']:
            guide_content += f"\n**Local-only features to replace:**\n"
            for feature in analysis['local_only_features']:
                guide_content += f"- `{feature}`\n"
                
        if analysis['required_changes']:
            guide_content += f"\n**Required changes:**\n"
            for change in analysis['required_changes']:
                guide_content += f"- {change}\n"
    
    # Add summary
    easy_files = sum(1 for a in analyses if a.get('migration_complexity') == 'easy')
    medium_files = sum(1 for a in analyses if a.get('migration_complexity') == 'medium')
    hard_files = sum(1 for a in analyses if a.get('migration_complexity') == 'hard')
    error_files = sum(1 for a in analyses if a.get('migration_complexity') == 'error')
    
    guide_content += f"""
## Migration Summary

- **Easy migrations**: {easy_files} files - minimal changes needed
- **Medium migrations**: {medium_files} files - import changes required
- **Hard migrations**: {hard_files} files - significant refactoring needed
- **Errors**: {error_files} files - could not analyze

## Tips for Successful Migration

1. **Start with easy files** - migrate files with no verl-lite specific code first
2. **Test incrementally** - migrate and test one component at a time
3. **Keep configs separate** - use different config files for local vs distributed
4. **Use feature flags** - conditionally enable Ray features based on environment

## Common Patterns

### Configuration Switching

```python
import os

if os.getenv('USE_RAY', 'False').lower() == 'true':
    # Full verl with Ray
    import ray
    from verl.trainer.ppo.ray_trainer import RayPPOTrainer
    ray.init()
    trainer = RayPPOTrainer(config)
else:
    # verl-lite local mode
    from verl_lite.trainer import LocalPPOTrainer
    trainer = LocalPPOTrainer(config)
```

### Rollout Engine Switching

```python
# Both versions can use the same server-mode rollout
if config.rollout_engine == 'vllm':
    # Works in both verl-lite and full verl
    from verl.workers.rollout.vllm_rollout import vLLMRollout
    rollout = vLLMRollout(config, server_mode=True)
```

## Troubleshooting

If you encounter issues during migration:

1. Check that all Ray workers are properly decorated with `@ray.remote`
2. Ensure Ray is initialized before creating remote objects
3. Verify that resource allocation matches your cluster configuration
4. Check that all imports point to the correct verl modules

For more help, see the full VERL documentation.
"""
    
    # Write the guide
    with open(output_file, 'w') as f:
        f.write(guide_content)
    
    print(f"Migration guide created: {output_file}")
    return guide_content


def create_migration_script(source_file: str, output_file: str):
    """Create a migrated version of a verl-lite file for full verl."""
    
    with open(source_file, 'r') as f:
        content = f.read()
    
    # Simple find-and-replace migrations
    migrations = {
        'verl_lite.trainer': 'verl.trainer.ppo',
        'verl_lite.workers': 'verl.workers',
        'verl_lite.utils': 'verl.utils',
        'LocalPPOTrainer': 'RayPPOTrainer',
        'LocalFSDPWorkers': 'FSDPWorkers',
        'LocalRolloutManager': 'RolloutManager',
        'mock_ray_remote': 'ray.remote',
        'mock_ray_get': 'ray.get',
    }
    
    migrated_content = content
    for old, new in migrations.items():
        migrated_content = migrated_content.replace(old, new)
    
    # Add Ray import if not present
    if 'import ray' not in migrated_content and ('ray.' in migrated_content or '@ray.remote' in migrated_content):
        migrated_content = 'import ray\n' + migrated_content
    
    with open(output_file, 'w') as f:
        f.write(migrated_content)
    
    print(f"Migrated file created: {output_file}")
    return migrated_content