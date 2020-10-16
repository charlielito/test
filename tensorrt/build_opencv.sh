# Jetson AGX Xavier
ARCH_BIN=7.2

# Value should be YES or NO
DOWNLOAD_OPENCV_EXTRAS=NO
# Source code directory
OPENCV_SOURCE_DIR=$HOME

WHEREAMI=$PWD

CLEANUP=true

sudo apt-get install -y \
    cmake \
    libavcodec-dev \
    libavformat-dev \
    libavutil-dev \
    libeigen3-dev \
    libglew-dev \
    libgtk2.0-dev \
    libgtk-3-dev \
    libjpeg-dev \
    libpng-dev \
    libpostproc-dev \
    libswscale-dev \
    libtbb-dev \
    libtiff5-dev \
    libv4l-dev \
    libxvidcore-dev \
    libx264-dev \
    qt5-default \
    zlib1g-dev \
    libgl1 \
    libglvnd-dev \
    pkg-config

# GStreamer support
sudo apt-get install -y libgstreamer1.0-dev libgstreamer-plugins-base1.0-dev 

# cd $OPENCV_SOURCE_DIR
# git clone https://github.com/opencv/opencv.git

# cd $OPENCV_SOURCE_DIR/opencv
# git clone https://github.com/opencv/opencv_contrib.git

cd $OPENCV_SOURCE_DIR/opencv
# mkdir build
cd build

# Directory in which to install opencv libraries
CMAKE_INSTALL_PREFIX=/usr/local

time cmake -D CMAKE_BUILD_TYPE=RELEASE \
      -D CMAKE_INSTALL_PREFIX=${CMAKE_INSTALL_PREFIX} \
      -D WITH_CUDA=ON \
      -D CUDA_ARCH_BIN=${ARCH_BIN} \
      -D CUDA_ARCH_PTX="" \
      -D ENABLE_FAST_MATH=ON \
      -D CUDA_FAST_MATH=ON \
      -D WITH_CUBLAS=ON \
      -D WITH_LIBV4L=ON \
      -D WITH_GSTREAMER=ON \
      -D WITH_GSTREAMER_0_10=OFF \
      -D WITH_QT=ON \
      -D WITH_OPENGL=OFF \
      -D CUDA_NVCC_FLAGS="--expt-relaxed-constexpr" \
      -D WITH_TBB=ON \
      -D PYTHON_EXECUTABLE=/usr/bin/python3 \
      -D OPENCV_EXTRA_MODULES_PATH=$OPENCV_SOURCE_DIR/opencv/opencv_contrib/modules \
      ../

if [ $? -eq 0 ] ; then
  echo "CMake configuration make successful"
else
  # Try to make again
  echo "CMake issues " >&2
  echo "Please check the configuration being used"
  exit 1
fi


NUM_CPU=$(nproc)
time make -j$(($NUM_CPU - 1))
if [ $? -eq 0 ] ; then
  echo "OpenCV make successful"
else
    # Try to make again
    echo "Make did not successfully build" >&2
    echo "Please fix issues and retry build"
    exit 1
fi

echo "Installing ... "
sudo make install
if [ $? -eq 0 ] ; then
   echo "OpenCV installed in: $CMAKE_INSTALL_PREFIX"
else
   echo "There was an issue with the final installation"
   exit 1
fi