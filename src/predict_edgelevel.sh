#!/bin/bash

GPU_ID=$2
export CUDA_VISIBLE_DEVICES=${GPU_ID}
export MODEL_NAME=google/electra-base-discriminator
export OMP_NUM_THREADS=3

PATH_MODEL=$1
FILE_VAL='../data/processed_dataset_edge_level/test-sents-amr.json'


python -u main_edgelevel.py --test --model_name_or_path ${MODEL_NAME} \
--test_data_file ${FILE_VAL} \
--batch_size 8 \
--save_dir ${PATH_MODEL}

