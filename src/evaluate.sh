#!/bin/bash

export OMP_NUM_THREADS=3
ROOT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"
NUMBER_GRAPHS=5
GPU_ID=$3
export CUDA_VISIBLE_DEVICES=${GPU_ID}
export MODEL_NAME=google/electra-base-discriminator
PREPROCESS_FOLDER=../data/preprocess

MODEL_TYPE=$1
JSON_FILE_VAL=$2
JSON_FILE_VAL=${JSON_FILE_VAL}

if [ "${MODEL_TYPE}" = "factgraph" ]; then
  PATH_MODEL='../checkpoints/factgraph'
else
  PATH_MODEL='../checkpoints/factgraph-edge'
fi


source ~/anaconda3/etc/profile.d/conda.sh

conda deactivate

conda activate preprocess-fatcgraph
python -u ${PREPROCESS_FOLDER}/preprocess_evaluate.py ${JSON_FILE_VAL} ${NUMBER_GRAPHS}
conda deactivate

conda activate factgraph
python -u evaluate.py --model_type ${MODEL_TYPE} --model_dir ${PATH_MODEL} --model_name_or_path ${MODEL_NAME} \
--test_data_file ${JSON_FILE_VAL}.processed
conda deactivate