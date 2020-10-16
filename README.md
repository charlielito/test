## In vanilla ubuntu Jetpack 4.3
Install depedencies
```
bash build_opencv.sh
bash build_tkdnn.sh
```

Build tensorRT optimized model
```
cd $HOME/tkDNN/build
export TKDNN_MODE=FP16
./test_yolo4
```

Run inference
```
cd $HOME/tkDNN/build
./demo yolo4_fp16.rt ../demo/yolo_test.mp4 y 80 1 0
```

### Results Xavier Ubuntu
```
Time stats:
Min: 24.2105 ms
Max: 46.2177 ms
Avg: 24.3411 ms 41.0828 FPS
```

## In balena
Build the image and after it has created the optimized model, login in and run:
```
cd /usr/src/app/tkDNN/build
./demo yolo4_fp16.rt ../demo/yolo_test.mp4 y 80 1 0
```
### Results Xavier Balena
```
Time stats:
Min: 45.3216 ms
Max: 85.8358 ms
Avg: 45.7216 ms 21.8715 FPS
```
