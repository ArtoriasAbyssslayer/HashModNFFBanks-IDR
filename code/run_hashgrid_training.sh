 #!/bin/bash
CUDA_LAUNCH_BLOCKING=1 OMP_NUM_THREADS=8 python.exe -u ./training/exp_runner.py --conf ./confs/embedder_conf_var/MultiResHash/dtu_fixed_cameras.conf --expname HashGrid --scan_id 65 --checkpoint latest --validation_slope_print --is_continue
