#!/usr/bin/env python
from pathlib import Path

from setuptools import find_packages, setup


def read_version() -> str:
    ns = {}
    version_file = Path(__file__).parent / "openhgnn" / "_version.py"
    exec(version_file.read_text(encoding="utf-8"), ns)
    return ns["__version__"]


install_requires = [
    "numpy>=1.26,<2.0",
    "pandas>=2.2,<3.0",
    "ogb>=1.3.6",
    "optuna>=4.0",
    "rdflib>=7.0",
    "colorama>=0.4.6",
    "colorlog>=6.0",
    "igraph>=0.11",
    "torch>=2.3,<=2.4.0",
    "dgl>=2.2,<=2.4.0",
    "TensorBoard>=2.0.0",
    "lmdb>=1.6",
    "ordered-set>=4.1",
    "scikit-learn>=1.3,<1.7",
]

classifiers = [
    "Development Status :: 4 - Beta",
    "License :: OSI Approved :: Apache Software License",
    "Programming Language :: Python :: 3",
    "Programming Language :: Python :: 3.10",
    "Programming Language :: Python :: 3.11",
    "Programming Language :: Python :: 3.12",
]

setup(
    name="openhgnn",
    version=read_version(),
    author="BUPT-GAMMA LAB",
    author_email="tyzhao@bupt.edu.cn",
    maintainer="Tianyu Zhao",
    license="Apache-2.0 License",
    description="An open-source toolkit for Heterogeneous Graph Neural Network",
    url="https://github.com/BUPT-GAMMA/OpenHGNN",
    download_url="https://github.com/BUPT-GAMMA/OpenHGNN",
    python_requires=">=3.10",
    packages=find_packages(),
    install_requires=install_requires,
    include_package_data=True,
    classifiers=classifiers,
    entry_points={
        "console_scripts": [
            "openhgnn=openhgnn.cli:main",
        ],
    },
)
