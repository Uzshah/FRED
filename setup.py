"""
Setup configuration for FRED package.
"""

from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="fred-depth",
    version="0.1.0",
    author="FRED Authors",
    author_email="",
    description="FRED: Fast Refinement for Equirectangular Depth",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/magus74/FRED",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Scientific/Engineering :: Image Processing",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
    ],
    python_requires=">=3.8",
    install_requires=[
        "numpy>=1.20.0",
        "torch>=1.10.0",
        "Pillow>=8.0.0",
        "matplotlib>=3.3.0",
        "transformers>=4.20.0",
        "promptda @ git+https://github.com/DepthAnything/PromptDA.git",
    ],
    extras_require={
        "dev": [
            "pytest>=6.0",
            "black>=21.0",
            "flake8>=3.9",
        ],
    },
    entry_points={
        "console_scripts": [
            "fred=fred.cli:main",
        ],
    },
)
