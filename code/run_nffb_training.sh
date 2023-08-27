#!/bin/bash
alias python=python3
OMP_NUM_THREADS=$(nproc) python3 -u  ./training/exp_runner.py --conf ./confs/embedder_conf_var/FFB/dtu_fixed_cameras.conf --expname NFFB_BUDA --scan_id 110 --checkpoint latest --validation_slope_print 
