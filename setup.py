#!/usr/bin/env python
# -*- coding: utf-8 -*-
from setuptools import setup, find_packages


install_requires = ['numpy>=1.16.5', 'pandas>=1.0.0', 'ogb>=1.1.0',
                    'torch>=1.8.1', 'dgl==0.7a210527', 'optuna']
setup_requires = []
tests_require = []

classifiers = [
    'Development Status :: 3 - Alpha',
    'License :: OSI Approved :: Apache Software License'
    'Programming Langugage :: Python :: 3.6',

]

setup(
    name = "openhgnn",
    version = "0.1.0",
    author = "BUPT-GAMMA LAB",
    author_email = "tyzhao@bupt.edu.cn",
    maintainer = "Tianyu Zhao",
    license = "Apache-2.0 License",

    description = "An open-source toolkit for Heterogeneous Graph Neural Network",
    long_description = " ",
    
    url = "https://github.com/BUPT-GAMMA/OpenHGNN",
    download_url = "https://github.com/BUPT-GAMMA/OpenHGNN",

    python_requires='>=3.6',

    packages = find_packages(),
    
    package_data = {
        '': ['*.ini'],
    },
    
    install_requires = install_requires,
    include_package_data = True,

    classifiers = classifiers
)
    
    
