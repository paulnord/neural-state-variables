#!/bin/bash

dataset=$1
gpu=$2

echo "========================================================================================"
echo "================ Training latentpred model on: $dataset (gpu id: $gpu) ================="
echo "========================================================================================"

screen -S train-"$dataset" -dm bash -c "CUDA_VISIBLE_DEVICES="$gpu" python3 ../main.py latentpred ../configs/"$dataset"_pmn/latentpred/config1.yaml ./logs_"$dataset"_encoder-decoder-64-pmn_1/lightning_logs/checkpoints/ ./logs_"$dataset"_refine-64_1/lightning_logs/checkpoints/; \
                                        CUDA_VISIBLE_DEVICES="$gpu" python3 ../main.py latentpred ../configs/"$dataset"_pmn/latentpred/config2.yaml ./logs_"$dataset"_encoder-decoder-64-pmn_2/lightning_logs/checkpoints/ ./logs_"$dataset"_refine-64_2/lightning_logs/checkpoints/; \
                                        CUDA_VISIBLE_DEVICES="$gpu" python3 ../main.py latentpred ../configs/"$dataset"_pmn/latentpred/config3.yaml ./logs_"$dataset"_encoder-decoder-64-pmn_3/lightning_logs/checkpoints/ ./logs_"$dataset"_refine-64_3/lightning_logs/checkpoints/; \
                                        exec sh";
