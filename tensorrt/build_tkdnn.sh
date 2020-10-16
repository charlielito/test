# Source code directory for tkDNN
CMAKE_SOURCE_DIR=$HOME
## ----- Build cmake 3.15 for tkDNN
sudo apt remove -y cmake
wget https://cmake.org/files/v3.15/cmake-3.15.0.tar.gz -O $CMAKE_SOURCE_DIR/cmake-3.15.0.tar.gz
cd $CMAKE_SOURCE_DIR
tar xf cmake-3.15.0.tar.gz
cd cmake-3.15.0
./configure
NUM_CPU=$(nproc)
sudo make install

# Install tkDNN
sudo apt install -y libyaml-cpp-dev
cd $CMAKE_SOURCE_DIR
git clone https://github.com/ceccocats/tkDNN.git
cd tkDNN
mkdir build
cd build
cmake .. 
make -j$(($NUM_CPU - 1))

# Compile yolov4 for FP16 precision
export TKDNN_MODE=FP16 
./test_yolo4
# ./demo yolo4_fp16.rt ../demo/yolo_test.mp4 y 80 1 0