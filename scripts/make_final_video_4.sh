#!/bin/bash

if [ $# -ge 1 ]
then
  dataset=$1
else
  echo "Usage: make_prediction_videos.sh <dataset> <encoder> <version>" 
  exit 1
fi

if [ $# -ge 2 ]
then
  encoder=$2
else
  encoder="encoder-decoder"
fi

if [ $# -ge 3 ]
then
  version=$3
else
  version="1"
fi

{
	pushd logs/logs_${dataset}_${encoder}_${version}/predictions
} || {
	echo "Failed to find dataset " $dataset
        exit 1
}


ffmpeg -y -i output0.mpg -i output1.mpg \
-i output2.mpg -i output3.mpg \
-filter_complex \
"[0:v][1:v]hstack=inputs=4[v]" \
-map "[v]" -vb 20M output_${dataset}_${encoder}_${version}.mpg


popd
