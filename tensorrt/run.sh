#!/bin/bash

# Enable all 8 cores
CORES=(
	# 0
	# 1
	# 2
	# 3
	4
	5
	6
	7 
)
for i in ${CORES[@]};
do 
	chcpu -e $i
done

cd /usr/src/app/tkDNN/build
FILE=yolo4_fp16.rt
export TKDNN_MODE=FP16

if test -f "$FILE"; then
    echo "$FILE exist"
else
    echo "$FILE no exist, creating one"
    ./test_yolo4
fi

sleep infinity 
