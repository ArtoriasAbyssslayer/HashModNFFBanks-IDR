#!/bin/bash
while true; do
    python -u ./training/exp_runner.py --conf  ./confs/embedder_conf_var/MultiResHashPointsAndViewDirs/dtu_fixed_cameras.conf  --expname 'HashGrid_Glossy_Buda' --scan_id 114 --checkpoint latest --validation_slope_print --is_continue
    EXIT_CODE=$?
    if [ $EXIT_CODE -eq 0 ]; then
        echo "Python script finished successfully, exiting loop."
        break
    else
        echo "Python script failed with exit code $EXIT_CODE, restarting..."
    fi
done