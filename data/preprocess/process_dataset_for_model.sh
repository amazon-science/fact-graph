#!/bin/bash

source ~/anaconda3/etc/profile.d/conda.sh

GPU_ID=$1
export CUDA_VISIBLE_DEVICES=${GPU_ID}

NUMBER_GRAPHS=5

conda deactivate
conda activate preprocess-fatcgraph

FOLDER=../processed_dataset

## CREATE SENTS

FILE_DATA=${FOLDER}/train.json
python 01_generate_sents_for_graphs.py ${FILE_DATA} ${NUMBER_GRAPHS}

FILE_DATA=${FOLDER}/dev.json
python 01_generate_sents_for_graphs.py ${FILE_DATA} ${NUMBER_GRAPHS}

FILE_DATA=${FOLDER}/test.json
python 01_generate_sents_for_graphs.py ${FILE_DATA} ${NUMBER_GRAPHS}


## EXTRACT SENTS

FILE_DATA=${FOLDER}/train-sents.json
python 02.1_get_amr_data.py ${FILE_DATA}

FILE_DATA=${FOLDER}/dev-sents.json
python 02.1_get_amr_data.py ${FILE_DATA}

FILE_DATA=${FOLDER}/test-sents.json
python 02.1_get_amr_data.py ${FILE_DATA}

conda deactivate
conda activate spring

### GENERATE AMRS
FOLDER_SPRING=spring
PATH_MODEL=${FOLDER_SPRING}/AMR3.parsing.pt


FILE_VAL=${FOLDER}/train-sents.txt
python -u ${FOLDER_SPRING}/bin/predict_amrs_from_plaintext.py --checkpoint ${PATH_MODEL} --texts ${FILE_VAL} --penman-linearization \
  --use-pointer-tokens > ${FILE_VAL}.amr

FILE_VAL=${FOLDER}/dev-sents.txt
python -u ${FOLDER_SPRING}/bin/predict_amrs_from_plaintext.py --checkpoint ${PATH_MODEL} --texts ${FILE_VAL} --penman-linearization \
  --use-pointer-tokens > ${FILE_VAL}.amr

FILE_VAL=${FOLDER}/test-sents.txt
python -u ${FOLDER_SPRING}/bin/predict_amrs_from_plaintext.py --checkpoint ${PATH_MODEL} --texts ${FILE_VAL} --penman-linearization \
  --use-pointer-tokens > ${FILE_VAL}.amr

conda deactivate

## GENERATE DATA FILES
conda activate preprocess-fatcgraph

AMR_DATA=${FOLDER}/train-sents.txt.amr
FILE_DATA=${FOLDER}/train-sents.json
python 02.2_create_amr_json_nopar.py ${FILE_DATA} ${AMR_DATA} ${NUMBER_GRAPHS}

AMR_DATA=${FOLDER}/dev-sents.txt.amr
FILE_DATA=${FOLDER}/dev-sents.json
python 02.2_create_amr_json_nopar.py ${FILE_DATA} ${AMR_DATA} ${NUMBER_GRAPHS}

AMR_DATA=${FOLDER}/test-sents.txt.amr
FILE_DATA=${FOLDER}/test-sents.json
python 02.2_create_amr_json_nopar.py ${FILE_DATA} ${AMR_DATA} ${NUMBER_GRAPHS}

conda deactivate


