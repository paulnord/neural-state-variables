#!/bin/bash

dataset=$1
gpu=$2

echo "==========================================================================================="
echo "============== Training encoder-decoder-64 model on: $dataset (gpu id: $gpu) =============="
echo "==========================================================================================="

screen -S train-"$dataset"-64 -dm bash -c "CUDA_VISIBLE_DEVICES="$gpu" python3 ../main.py ../configs/"$dataset"_pmn/model64/config1.yaml; \
                                           CUDA_VISIBLE_DEVICES="$gpu" python3 ../main.py ../configs/"$dataset"_pmn/model64/config2.yaml; \
                                           CUDA_VISIBLE_DEVICES="$gpu" python3 ../main.py ../configs/"$dataset"_pmn/model64/config3.yaml; \
                                           exec sh";
