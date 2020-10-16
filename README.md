# Test Tensorflow model
## In vanilla ubuntu Jetpack 4.3
You need to install tensorflow 2.1. Easily with: `pip3 install --pre --extra-index-url https://developer.download.nvidia.com/compute/redist/jp/v43 tensorflow==2.1.0`.

Then just execute

```
python3 tensorflow/model/bisenetv2_tensorflow_depthconv.py
```

### Results Xavier Ubuntu
```
FPS 39.854563242417385
Time 0.025091229677200316
```
## In balena
Build the application inside the `tensorflow` directory.

### Results Xavier Balena
```
FPS 19.597094896866313
Time 0.05102797150611878
```
# Test TensorRT
## In vanilla ubuntu Jetpack 4.3
Install depedencies inside the `tensorrt` directory.
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
Build the image inside the `tensorrt` directory and after it has created the optimized model, login in and run:
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
