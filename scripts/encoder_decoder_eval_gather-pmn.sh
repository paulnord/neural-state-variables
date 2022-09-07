#!/bin/bash

dataset=$1
gpu=$2

echo "======================================================================================================"
echo "============== Evaluating (Gathering) encoder-decoder model on: $dataset (gpu id: $gpu) =============="
echo "======================================================================================================"

screen -S eval-"$dataset" -dm bash -c "CUDA_VISIBLE_DEVICES="$gpu" python3 ../eval.py ../configs/"$dataset"_pmn/model/config1.yaml ./logs/logs_"$dataset"_encoder-decoder-pmn_1/lightning_logs/checkpoints NA eval-train NA; \
                                       CUDA_VISIBLE_DEVICES="$gpu" python3 ../eval.py ../configs/"$dataset"_pmn/model/config2.yaml ./logs/logs_"$dataset"_encoder-decoder-pmn_2/lightning_logs/checkpoints NA eval-train NA; \
                                       CUDA_VISIBLE_DEVICES="$gpu" python3 ../eval.py ../configs/"$dataset"_pmn/model/config3.yaml ./logs/logs_"$dataset"_encoder-decoder-pmn_3/lightning_logs/checkpoints NA eval-train NA; \
                                       exec sh";
