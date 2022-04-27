#!/bin/bash

source ~/anaconda3/etc/profile.d/conda.sh

GPU_ID=$1
export CUDA_VISIBLE_DEVICES=${GPU_ID}

NUMBER_GRAPHS=5
AMR_PARSER='amrlib'

conda deactivate
conda activate preprocess-fatcgraph

MAYNEZ_FILE=../maynez/hallucination_annotations_xsum_summaries.csv
FOLDER=../processed_dataset_edge_level
mkdir -p ${FOLDER}

python 00_convert_edge_level_dataset.py ../edge_level_data ${FOLDER}

## CREATE SENTS

FILE_DATA=${FOLDER}/train.json
python 01_generate_sents_for_graphs.py ${FILE_DATA} ${NUMBER_GRAPHS}

FILE_DATA=${FOLDER}/test.json
python 01_generate_sents_for_graphs.py ${FILE_DATA} ${NUMBER_GRAPHS}

conda deactivate

if [ "${AMR_PARSER}" = "spring" ]; then

  conda activate preprocess-fatcgraph

  ## EXTRACT SENTS

  FILE_DATA=${FOLDER}/train-sents.json
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

  FILE_VAL=${FOLDER}/test-sents.txt
  python -u ${FOLDER_SPRING}/bin/predict_amrs_from_plaintext.py --checkpoint ${PATH_MODEL} --texts ${FILE_VAL} --penman-linearization \
    --use-pointer-tokens > ${FILE_VAL}.amr

  conda deactivate

  conda activate preprocess-fatcgraph

  AMR_DATA=${FOLDER}/train-sents.txt.amr
  python 02.1.1_align_amrs.py $AMR_DATA

  AMR_DATA=${FOLDER}/test-sents.txt.amr
  python 02.1.1_align_amrs.py $AMR_DATA

  AMR_DATA=${FOLDER}/train-sents.txt.amr.align
  FILE_DATA=${FOLDER}/train-sents.json
  python 02.2_create_amr_json_nopar_edge_level.py ${FILE_DATA} ${AMR_DATA} ${NUMBER_GRAPHS} ${MAYNEZ_FILE}

  AMR_DATA=${FOLDER}/test-sents.txt.amr.align
  FILE_DATA=${FOLDER}/test-sents.json
  python 02.2_create_amr_json_nopar_edge_level.py ${FILE_DATA} ${AMR_DATA} ${NUMBER_GRAPHS} ${MAYNEZ_FILE}

  conda deactivate
else
  conda activate preprocess-fatcgraph

  FILE_VAL=${FOLDER}/train-sents.json
  python -u 02_generate_amrs.py ${FILE_VAL} ${FILE_VAL}.amr

  FILE_VAL=${FOLDER}/test-sents.json
  python -u 02_generate_amrs.py ${FILE_VAL} ${FILE_VAL}.amr

  AMR_DATA=${FOLDER}/train-sents.json.amr
  python 02.1.1_align_amrs.py $AMR_DATA

  AMR_DATA=${FOLDER}/test-sents.json.amr
  python 02.1.1_align_amrs.py $AMR_DATA

  AMR_DATA=${FOLDER}/train-sents.json.amr.align
  FILE_DATA=${FOLDER}/train-sents.json
  python 02.2_create_amr_json_nopar_edge_level.py ${FILE_DATA} ${AMR_DATA} ${NUMBER_GRAPHS} ${MAYNEZ_FILE}

  AMR_DATA=${FOLDER}/test-sents.json.amr.align
  FILE_DATA=${FOLDER}/test-sents.json
  python 02.2_create_amr_json_nopar_edge_level.py ${FILE_DATA} ${AMR_DATA} ${NUMBER_GRAPHS} ${MAYNEZ_FILE}

  conda deactivate
fi




