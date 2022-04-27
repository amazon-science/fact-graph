#!/bin/bash

FILE_TRAIN='../data/processed_dataset/train-sents-amr.json'
FILE_VAL='../data/processed_dataset/dev-sents-amr.json'
FILE_TEST='../data/processed_dataset/test-sents-amr.json'

GPU_ID=$1
export CUDA_VISIBLE_DEVICES=${GPU_ID}
export MODEL_NAME=google/electra-base-discriminator
export OMP_NUM_THREADS=3

NAME_EXECUTION=$MODEL_NAME-$RANDOM
PATH_MODEL=checkpoints/${NAME_EXECUTION}
PRE_MODEL_GRAPH_ADAPT=../checkpoints/graph_adapters.bin
PRE_MODEL_TEXT_ADAPT=../checkpoints/text_adapters.bin
rm -rf ${PATH_MODEL}
mkdir -p ${PATH_MODEL}
python -u main.py --model_name_or_path ${MODEL_NAME} \
--train_data_file ${FILE_TRAIN} \
--val_data_file ${FILE_VAL} \
--test_data_file ${FILE_TEST} \
--save_every_k_step 300 \
--batch_size 8 \
--adapter_size 32 \
--num_epoch 4 \
--pretrained_model_adapters ${PRE_MODEL_TEXT_ADAPT} \
--pretrained_model_graph_adapters ${PRE_MODEL_GRAPH_ADAPT} \
--save_dir ${PATH_MODEL}

