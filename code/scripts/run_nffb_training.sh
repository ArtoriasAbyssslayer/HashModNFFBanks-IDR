#!/bin/bash
cd ..
CUDA_LAUNCH_BLOCKING=0 TORCH_USE_DSA=1 OMP_NUM_THREADS=$(nproc) python3 -u  ./training/exp_runner.py --conf ./confs/embedder_conf_var/FFB/dtu_fixed_cameras.conf --expname NFFB_glossy_buda --scan_id 118 --checkpoint latest --validation_slope_print --is_continue
