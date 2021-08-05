#!/bin/bash

_log_dir=$1
_log_name=$2
kernprof -l -v -u 1 test_MAGNN_sampler.py >logs/$_log_dir/$_log_name 2>&1 &
