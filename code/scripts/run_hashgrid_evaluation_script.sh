#!/bin/bash
# python3 -u ./evaluation/eval --exps_folder exps --expname [EncodingNetName] --conf [EncodingNetConfigVariation] --scan_id [SCAN_ID] --checkpoint [latest] [--eval_rendering]
cd ..
alias python=python3
CUDA_LAUNCH_BLOCKING=1 TORCH_USE_CUDA_DSA=1 python -u ./evaluation/eval.py --exps_folder exps --expname HashGrid --conf ./confs/embedder_conf_var/MultiResHash/dtu_fixed_cameras.conf --scan_id 65 --checkpoint latest --eval_rendering
