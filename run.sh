#!/bin/bash
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