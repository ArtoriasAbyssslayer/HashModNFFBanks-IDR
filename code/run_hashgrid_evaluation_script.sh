#!/bin/bash
# python3 -u ./evaluation/eval --exps_folder exps --expname [EncodingNetName] --conf [EncodingNetConfigVariation] --scan_id [SCAN_ID] --checkpoint [latest] [--eval_rendering]
python3 -u ./evaluation/eval.py --exps_folder exps --expname HashGrid --conf ./confs/embedder_conf_var/MultiResHash/dtu_fixed_cameras.conf --scan_id 65 --checkpoint latest --eval_rendering
