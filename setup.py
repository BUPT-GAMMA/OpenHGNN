#!/usr/bin/env python
# -*- coding: utf-8 -*-
from pathlib import Path

from setuptools import find_packages, setup


ROOT = Path(__file__).parent
version_ns = {}
exec((ROOT / "openhgnn" / "_version.py").read_text(encoding="utf-8"), version_ns)

install_requires = [
    "numpy>=1.16.5",
    "pandas>=1.0.0",
    "ogb>=1.3.1",
    "torch>=2.3,<2.6",
    "dgl>=2.2,<2.6",
    "optuna>=3.0",
    "colorama",
    "TensorBoard>=2.0.0",
    "lmdb",
    "ordered_set",
    "rdflib",
    "igraph",
]

classifiers = [
    "Development Status :: 3 - Alpha",
    "License :: OSI Approved :: Apache Software License",
    "Programming Language :: Python :: 3",
    "Programming Language :: Python :: 3.10",
    "Programming Language :: Python :: 3.11",
    "Programming Language :: Python :: 3.12",
]

setup(
    name="openhgnn",
    version=version_ns["__version__"],
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
    entry_points={"console_scripts": ["openhgnn=openhgnn.cli:main"]},
)
