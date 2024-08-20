#!/usr/bin/env python
# -*- coding: utf-8 -*-
from setuptools import setup, find_packages

install_requires = ['numpy>=1.16.5', 'pandas>=1.0.0', 'ogb>=1.1.0',
                    'torch>=2.2.1', 'optuna', 'colorama', 'TensorBoard>=2.0.0']
setup_requires = []
tests_require = []

classifiers = [
    'Development Status :: 3 - Alpha',
    'License :: OSI Approved :: Apache Software License',
    'Programming Language :: Python :: 3',
]

setup(
    name="openhgnn",
    version="0.7.0",
    author="BUPT-GAMMA LAB",
    author_email="tyzhao@bupt.edu.cn",
    maintainer="Tianyu Zhao",
    license="Apache-2.0 License",
    description="An open-source toolkit for Heterogeneous Graph Neural Network",
    url="https://github.com/BUPT-GAMMA/OpenHGNN",
    download_url="https://github.com/BUPT-GAMMA/OpenHGNN",
    python_requires='>=3.6',
    packages=find_packages(),
    install_requires=install_requires,
    include_package_data=True,
    classifiers=classifiers
)
