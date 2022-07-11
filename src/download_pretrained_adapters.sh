#!/bin/bash

GRAPH_ADAPTERS=https://public.ukp.informatik.tu-darmstadt.de/ribeiro/factgraph/graph_adapters.bin
TEXT_ADAPTERS=https://public.ukp.informatik.tu-darmstadt.de/ribeiro/factgraph/text_adapters.bin

mkdir -p ../checkpoints
wget ${GRAPH_ADAPTERS} -P ../checkpoints
wget ${TEXT_ADAPTERS} -P ../checkpoints




