#!/bin/bash
alias python=python3
CUDA_LAUNCH_BLOCKING=1 TORCH_USE_CUDA_DSA=1 python3 -u  ./training/exp_runner.py --conf ./confs/embedder_conf_var/FFB/dtu_fixed_cameras.conf --expname NeuralFFB_ --scan_id 65 --checkpoint latest --validation_slope_print
