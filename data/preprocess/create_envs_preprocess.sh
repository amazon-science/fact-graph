#!/bin/bash

source ~/anaconda3/etc/profile.d/conda.sh

conda create -n preprocess-fatcgraph python=3.8
conda activate preprocess-fatcgraph
pip install torch==1.9.0+cu111 torchvision==0.10.0+cu111 torchaudio==0.9.0 -f https://download.pytorch.org/whl/torch_stable.html
pip install -r requirements-preprocess.txt
git clone https://github.com//ablodge/amr-utils
pip install penman
pip install ./amr-utils
wget https://github.com/bjascob/amrlib-models/releases/download/parse_xfm_bart_large-v0_1_0/model_parse_xfm_bart_large-v0_1_0.tar.gz
tar zxvf model_parse_xfm_bart_large-v0_1_0.tar.gz

git clone https://github.com/clab/fast_align.git
cd fast_align
mkdir build
cd build
cmake ..
make
cd ../../

conda deactivate

conda create -n spring python=3.8
conda activate spring
conda install pytorch==1.5.0 torchvision cudatoolkit=10.2 -c pytorch
git clone https://github.com/SapienzaNLP/spring.git
cd spring
wget http://nlp.uniroma1.it/AMR/AMR3.parsing-1.0.tar.bz2
tar -xf AMR3.parsing-1.0.tar.bz2
pip install -r requirements.txt
cp ../scripts/predict_amrs_from_plaintext.py bin/
pip install -e .

conda deactivate


