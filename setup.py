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

"""Setup script for verl-lite."""

from setuptools import setup, find_packages
import os

# Read README for long description
with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

# Core dependencies that mirror verl but without Ray
install_requires = [
    # Core verl dependency - this is the key requirement!
    "verl==0.5.0",  # Must have full verl installed to import from it
    
    # Required for DataProto batch operations
    "tensordict>=0.3.0",
    
    # HTTP client for server-mode rollout
    "aiohttp>=3.8.0",
    "requests>=2.25.0",
    
    # Optional: for development and examples
    "pandas>=1.3.0",
    "pyarrow>=5.0.0",
]

# Optional dependencies for different rollout engines
extras_require = {
    "vllm": [
        "vllm>=0.7.3",
    ],
    "sglang": [
        "sglang>=0.4.0",  
    ],
    "dev": [
        "pytest>=7.0.0",
        "pytest-asyncio>=0.21.0",
        "pre-commit>=3.0.0",
        "ruff>=0.1.0",
        "mypy>=1.0.0",
    ],
    "all": [
        "vllm>=0.7.3",
        "sglang>=0.4.0", 
        "pytest>=7.0.0",
        "pytest-asyncio>=0.21.0",
    ],
}

setup(
    name="verl-lite",
    version="0.1.0",
    author="Zhongwen Xu",
    author_email="zhongwenxu@tencent.com",
    description="Ray-free version of VERL for single-machine debugging and prototyping",
    long_description=long_description,
    long_description_content_type="text/markdown",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: Apache Software License", 
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Software Development :: Libraries :: Python Modules",
    ],
    python_requires=">=3.10",
    install_requires=install_requires,
    extras_require=extras_require,
    include_package_data=True,
    package_data={
        "verl_lite": ["recipe/**/*"],
    },
    entry_points={
        "console_scripts": [
            "verl-lite-grpo=verl_lite.recipe.grpo.train_grpo:main",
        ],
    },
)