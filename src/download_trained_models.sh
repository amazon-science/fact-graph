#!/bin/bash

GRAPH_ADAPTERS=https://public.ukp.informatik.tu-darmstadt.de/ribeiro/factgraph/factgraph.tar.gz
TEXT_ADAPTERS=https://public.ukp.informatik.tu-darmstadt.de/ribeiro/factgraph/factgraph-edge.tar.gz

mkdir -p ../checkpoints
wget ${GRAPH_ADAPTERS} -P ../checkpoints
wget ${TEXT_ADAPTERS} -P ../checkpoints


tar zxvf ../checkpoints/factgraph.tar.gz -C ../checkpoints/
tar zxvf ../checkpoints/factgraph-edge.tar.gz -C ../checkpoints/

wget https://github.com/bjascob/amrlib-models/releases/download/parse_xfm_bart_large-v0_1_0/model_parse_xfm_bart_large-v0_1_0.tar.gz -P ../data/preprocess/
tar zxvf ../data/preprocess/model_parse_xfm_bart_large-v0_1_0.tar.gz -C ../data/preprocess/