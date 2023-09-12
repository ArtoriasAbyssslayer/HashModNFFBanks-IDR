#!/bin/bash
cd ..
alias python=python3
CUDA_LAUNCH_BLOCKING=0 TORCH_USE_CUDA_DSA=1 python -u ./evaluation/eval.py --exps_folder exps --expname HashGrid --conf .confs/embedder_conf_var/MultiResHashPointsPosencViews --scan_id 65 --checkpoint latest --eval_rendering