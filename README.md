## In vanilla ubuntu Jetpack 4.3
Install depedencies
```
bash build_opencv.sh
bash build_tkdnn.sh
```
Run inference
```
cd $HOME/tkDNN/build
./demo yolo4_fp16.rt ../demo/yolo_test.mp4 y 80 1 0
```

## For balena
Build the image and after it has created the optimized model, login in and run:
```
cd /usr/src/app/tkDNN/build
./demo yolo4_fp16.rt ../demo/yolo_test.mp4 y 80 1 0
```