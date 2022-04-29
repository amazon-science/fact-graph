#!/bin/bash

FILE_CNNDM_QAGS=https://github.com/W4ngatang/qags/raw/master/data/mturk_cnndm.jsonl
FILE_XSUM_QAGS=https://github.com/W4ngatang/qags/raw/master/data/mturk_xsum.jsonl

FILE_FACTCC=https://storage.googleapis.com/sfr-factcc-data-research/unpaired_annotated_data.tar.gz

FILE_MAYNEZ=https://github.com/google-research-datasets/xsum_hallucination_annotations/raw/master/hallucination_annotations_xsum_summaries.csv

FILE_FRANK=https://github.com/artidoro/frank/raw/main/data/human_annotations_sentence.json

rm -rf qags
mkdir -p qags
wget ${FILE_CNNDM_QAGS} -P qags
wget ${FILE_XSUM_QAGS} -P qags

rm -rf factcc
mkdir -p factcc
wget ${FILE_FACTCC} -P factcc
tar zxvf factcc/unpaired_annotated_data.tar.gz -C factcc/

rm -rf maynez
mkdir -p maynez
wget ${FILE_MAYNEZ} -P maynez

rm -rf frank
mkdir -p frank
wget ${FILE_FRANK} -P frank


python generate_dataset_qags.py qags/mturk_cnndm.jsonl qags/mturk_xsum.jsonl qags/processed.json

python generate_dataset_factcc.py factcc/unpaired_annotated_data/ factcc/processed.json

python generate_dataset_maynez.py maynez/hallucination_annotations_xsum_summaries.csv maynez/processed.json

python generate_dataset_frank.py frank/human_annotations_sentence.json frank/processed.json

rm -rf processed_dataset
mkdir -p processed_dataset

python generate_consolidated_dataset.py
