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


## Fix the filenames which contain single digits (eg. 44_2.png -> 44_02.png)
RUNS=`ls *_?.png *_?.jpg`

for i in ${RUNS}
  do
        echo echo Entry: $i
        split1=`echo $i | cut -f1 -d"_" `
        split2=`echo $i | cut -f2 -d"_" `
        fullCMD="mv "${i}" "${split1}"_0"${split2}
        `$fullCMD`
done

RUNS=`ls *_??.png *_??.jpg`

for i in ${RUNS}
  do
        echo echo Entry: $i
        split1=`echo $i | cut -f1 -d"_" `
        split2=`echo $i | cut -f2 -d"_" `
        fullCMD="mv "${i}" "${split1}"_0"${split2}
        `$fullCMD`
done

RUNS=`ls *_???.png *_???.jpg`

for i in ${RUNS}
  do
        echo echo Entry: $i
        split1=`echo $i | cut -f1 -d"_" `
        split2=`echo $i | cut -f2 -d"_" `
        fullCMD="mv "${i}" "${split1}"_0"${split2}
        `$fullCMD`
done


## Create videos for each set of images

RUNS=`ls *png *jpg`

arrVar=()
for i in ${RUNS}
  do
	split1=`echo $i | cut -f1 -d"_" `
	#echo $split1
	x=false
	for i in "${arrVar[@]}"
        do
            if [ "$i" == "$split1" ] ; then
		x=true
		continue
            fi
        done
	
	if ! $x
	then
	   #echo $split1
	   arrVar+=($split1)
	fi
done

echo echo "got list of files " $arrVar

for i in ${arrVar[@]}
  do
        echo echo creating: ${i}.mpg
	fullCMD="convert -delay 1 "${i}_*" "${i}.mpg
        `$fullCMD`
done


## merge videos into side-by-side single video

RUNS=`ls [0-9]*mpg`

arrVar=()
for i in ${RUNS}
  do
	split1=`echo $i `
	#echo $split1
	x=false
	for j in "${arrVar[@]}"
        do
            if [ "$j" == "$split1" ] ; then
		x=true
		continue
            fi
        done
	
	if ! $x
	then
	   #echo $split1
	   arrVar+=($split1)
	fi
done

echo echo "got list of files " ${arrVar[@]}

numFiles=${#arrVar[@]}
numGroups=$(((${numFiles}+7)/8))
echo "num Files :" $numFiles 
echo "num Groups :" $numGroups

for ((x=0;x<8;x++)); do
	i=$(($x*${numGroups}))
	echo $i
	#echo "sub array " ${arrVar[@]:${i}:8}
	subArray=""${arrVar[@]:${i}:${numGroups}}
	group=`echo ${subArray// /|}`
	echo ${group}
	outputFile="output"${x}".mpg"
	ffmpeg -y -vb 20M -i concat:"${group}" ${outputFile}
done

ffmpeg -y -i output0.mpg -i output1.mpg \
-i output2.mpg -i output3.mpg \
-i output4.mpg -i output5.mpg \
-i output6.mpg -i output7.mpg \
-filter_complex \
"[0:v][1:v]hstack=inputs=8[v]" \
-map "[v]" -vb 20M output_${dataset}_${encoder}_${version}.mpg


popd
