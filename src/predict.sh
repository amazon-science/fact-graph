#!/bin/bash

export OMP_NUM_THREADS=3

GPU_ID=$2
export CUDA_VISIBLE_DEVICES=${GPU_ID}
export MODEL_NAME=google/electra-base-discriminator

PATH_MODEL=$1
FILE_VAL='../data/processed_dataset/test-sents-amr.json'

CUDA_LAUNCH_BLOCKING=1 python -u main.py --test --save_dir ${PATH_MODEL} --model_name_or_path ${MODEL_NAME} \
--test_data_file ${FILE_VAL}