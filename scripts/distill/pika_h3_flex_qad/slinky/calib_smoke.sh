#!/bin/bash
# Calibration smoke on one tray: 2 collect rows, candidates, 2 MSE rows + outliers, choose.
R=/mnt/nfs/vlm-aryan/pika-flex-qad; O=$R/runs/calib-smoke; rm -rf $O
bash $R/jobs/calib.sh collect $R/smoke-ref2va.yaml $O 0 1 --max-rows 2 || exit 1
bash $R/jobs/calib.sh candidates - $O --outlier-rows 1 || exit 1
bash $R/jobs/calib.sh mse $R/smoke-ref2va.yaml $O 0 1 --mse-rows-per-group 1 || exit 1
bash $R/jobs/calib.sh choose - $O || exit 1
ls -la $O
