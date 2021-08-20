#!/bin/bash

_log_dir='dblp'
_log_name='test'

nohup python -u test_MAGNN_sampler.py >logs/$_log_dir/$_log_name 2>&1 &