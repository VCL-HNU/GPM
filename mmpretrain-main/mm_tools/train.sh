#!/usr/bin/env bash

cd "$(dirname "$0")/.." || exit
BASE_PATH=$(pwd)
echo "Current working directory: $BASE_PATH"
output_folder_parent="$BASE_PATH/Logs"

export CUDA_VISIBLE_DEVICES=0


CONFIG="$BASE_PATH/configs/GPM/vits_baseline_wo_pretrain.py"

current_time=$(date "+%Y%m%d%H%M%S")
output_folder="$output_folder_parent/Logs_$current_time"
mkdir -p "$output_folder"
PYTHONPATH="$BASE_PATH":$PYTHONPATH \
python $BASE_PATH/mm_tools/train.py $CONFIG \
--work-dir "$output_folder" \
2>&1 | tee -a "$output_folder/output.log"
