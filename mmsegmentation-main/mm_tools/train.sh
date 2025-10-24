#!/bin/bash

cd "$(dirname "$0")/.." || exit
BASE_PATH=$(pwd)
echo "Current working directory: $BASE_PATH"
output_folder_parent="$BASE_PATH/Logs"
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:32
export CUDA_VISIBLE_DEVICES=5


CONFIG="$BASE_PATH/configs/GPM/SegNeXt-tiny-20k-baseline.py"

current_time=$(date "+%Y%m%d%H%M%S")
output_folder="$output_folder_parent/Logs_$current_time"
mkdir -p "$output_folder"
PYTHONPATH="$BASE_PATH":$PYTHONPATH \
python $BASE_PATH/mm_tools/train.py $CONFIG \
--work-dir "$output_folder" \
2>&1 | tee -a "$output_folder/output.log"
