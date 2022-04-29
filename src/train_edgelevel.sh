#!/bin/bash

FILE_TRAIN='../data/processed_dataset_edge_level/train-sents-amr.json'
FILE_VAL='../data/processed_dataset_edge_level/test-sents-amr.json'

GPU_ID=$1
export CUDA_VISIBLE_DEVICES=${GPU_ID}
export MODEL_NAME=google/electra-base-discriminator
export OMP_NUM_THREADS=3

NAME_EXECUTION=$MODEL_NAME-$RANDOM
PATH_MODEL=../checkpoints/${NAME_EXECUTION}

rm -rf ${PATH_MODEL}
mkdir -p ${PATH_MODEL}
python -u main_edgelevel.py --model_name_or_path ${MODEL_NAME} \
--train_data_file ${FILE_TRAIN} \
--val_data_file ${FILE_VAL} \
--save_every_k_step 100 \
--batch_size 8 \
--adapter_size 32 \
--num_epoch 2 \
--save_dir ${PATH_MODEL}

