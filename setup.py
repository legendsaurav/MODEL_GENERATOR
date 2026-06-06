"""
MODEL_GENERATOR_V2 — Package installation script.

Ultra-quality single-image-to-3D-mesh generation system
based on Tencent Hunyuan3D-2.1 (geometry-only).
"""

from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as f:
    long_description = f.read()

with open("requirements.txt", "r", encoding="utf-8") as f:
    requirements = [
        line.strip()
        for line in f
        if line.strip() and not line.startswith("#")
    ]

setup(
    name="MODEL_GENERATOR_V2",
    version="2.0.0",
    author="MODEL_GENERATOR_V2",
    description=(
        "Ultra-quality single-image-to-3D-mesh generation "
        "based on Hunyuan3D-2.1"
    ),
    long_description=long_description,
    long_description_content_type="text/markdown",
    packages=find_packages(exclude=["tests", "tests.*", "outputs"]),
    python_requires=">=3.10",
    install_requires=requirements,
    entry_points={
        "console_scripts": [
            "model-gen-v2=MODEL_GENERATOR_V2.generate:main",
        ],
    },
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
    ],
    include_package_data=True,
    package_data={
        "MODEL_GENERATOR_V2": ["configs/*.yaml"],
    },
)
