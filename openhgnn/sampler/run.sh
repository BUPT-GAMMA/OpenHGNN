#!/bin/bash

_log_dir=$1
_log_name=$2

nohup python -u test_MAGNN_sampler.py >logs/$_log_dir/$_log_name 2>&1 &