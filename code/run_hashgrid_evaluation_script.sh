#!/bin/bash
python3 ./evaluation/eval.py --exps_folder exps --expname HashGrid --conf ./confs/embedder_conf_var/MultiResHash/dtu_fixed_cameras.conf --scan_id 65 --checkpoint latest --eval_rendering
