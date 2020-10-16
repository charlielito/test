
FROM balenalib/jetson-xavier-ubuntu:bionic

ENV DEBIAN_FRONTEND noninteractive

RUN apt-get update && apt-get install --no-install-recommends -y \
    apt-utils \
    bzip2 \
    lbzip2 \ 
    tar \
    wget \
    zlib1g-dev \
    libcurl4-gnutls-dev \
    python3-dev \
    python3-numpy \
    python3-pip \
    libyaml-cpp-dev \
    bzip2 \
    xorg \
    && apt autoremove && apt clean -y \
    && rm -rf /var/lib/apt/lists/*


WORKDIR /usr/src/app
COPY download.sh .
RUN bash download.sh \
    && dpkg -i cuda-repo-l4t-10-0-local-10.0.326_1.0-1_arm64.deb \
    libcudnn7_7.6.3.28-1+cuda10.0_arm64.deb \
    libcudnn7-dev_7.6.3.28-1+cuda10.0_arm64.deb \
    && apt-key add /var/cuda-repo-10-0-local-10.0.326/7fa2af80.pub \
    && apt-get update && apt-get install --no-install-recommends -y \
    cuda-toolkit-10-0 cuda-libraries-dev-10-0 \
    && rm -rf *.deb \
    && tar xjf nvidia_drivers.tbz2 -C / \
    && tar xjf config.tbz2 -C / --exclude=etc/hosts --exclude=etc/hostname \
    && echo "/usr/lib/aarch64-linux-gnu/tegra" > /etc/ld.so.conf.d/nvidia-tegra.conf \
    && ldconfig \
    && tar xvf nv_tools.tbz2 -C / --exclude=/home \
    && tar xvf nvgstapps.tbz2 -C / \
    && tar -C /usr/src/app --strip-components=2 -xvf nv_tools.tbz2 usr/bin/ \
    && ln -s /etc/nvpmodel/nvpmodel_t186.conf /etc/nvpmodel.conf \
    && rm *.tbz2 \
    && apt autoremove && apt clean -y \
    && rm -rf /var/lib/apt/lists/*
# -----------------------------------------------------------------------------
# Install tensorRT stuff
WORKDIR /L4T/dpkg
COPY tensorRT.txt .
RUN wget -qi tensorRT.txt \
    && dpkg -R --install /L4T/dpkg/ \ 
    && rm -rf /var/lib/apt/lists/* \
    && rm -rf /L4T

# -----------------------------------------------------------------------------
# Install cmake from source, because latest version not available in apt repos
ENV CMAKE_SOURCE_DIR=/cmake_build
# ----- Cmake 3.15 is required for tkDNN library (later on)
RUN mkdir $CMAKE_SOURCE_DIR \
    && wget https://cmake.org/files/v3.15/cmake-3.15.0.tar.gz -O $CMAKE_SOURCE_DIR/cmake-3.15.0.tar.gz \
    && cd $CMAKE_SOURCE_DIR \
    && tar xf cmake-3.15.0.tar.gz \
    && cd cmake-3.15.0  \
    # install with curl to support ssl urls
    && ./bootstrap --parallel=$(nproc) --system-curl \
    && make -j $(nproc) \
    && make install \
    && rm -r $CMAKE_SOURCE_DIR


WORKDIR /usr/src/app
# OpenCV Installation with contrib
ENV OPENCV_VERSION=4.4.0  \
    OPENCV_CONTRIB_VERSION=4.4.0  \
    OPENCV_LOG_LEVEL=ERROR
RUN apt update && apt install --no-install-recommends -y \
    # ------------------------------
    # Generic tools
    build-essential \
    git \
    pkg-config \
    unzip \
    yasm \
    checkinstall \
    # ------------------------------
    # Image I/O libs
    libjpeg-dev \
    libpng-dev \
    libtiff-dev \
    # ------------------------------
    # Video/Audio Libs - FFMPEG, GSTREAMER, x264 and so on.
    libavcodec-dev \
    libavformat-dev \
    libswscale-dev \
    # libavresample \
    libgstreamer1.0-dev \
    libgstreamer-plugins-base1.0-dev \
    libxvidcore-dev \
    x264 \
    libx264-dev \
    libfaac-dev \
    libmp3lame-dev \
    libtheora-dev \
    libvorbis-dev \
    # Cameras programming interface libs
    libdc1394-22 \
    libdc1394-22-dev \
    libxine2-dev \
    libv4l-dev \
    v4l-utils \
    # Ohters
    openexr \
    # ------------------------------
    # Parallelism library C++ for CPU
    libtbb-dev \
    # ------------------------------
    # Optimization libraries for OpenCV
    libatlas-base-dev \
    gfortran \
    # ------------------------------
    && apt autoremove && apt clean -y \
    && rm -rf /var/lib/apt/lists/*

# Jetson AGX Xavier
ENV ARCH_BIN 7.2
RUN \
    # Downlaoding and extracting OpenCV
    mkdir ~/opencv_build \
    && cd ~/opencv_build \
    && git clone -q --branch $OPENCV_VERSION https://github.com/opencv/opencv.git \
    && git clone -q --branch $OPENCV_CONTRIB_VERSION https://github.com/opencv/opencv_contrib.git \
    && cd ~/opencv_build/opencv \
    && mkdir build && cd build \
    # ------------------------------
    && cmake -D CMAKE_BUILD_TYPE=RELEASE \
    -D WITH_CUDA=ON \
    -D WITH_CUBLAS=ON \ 
    -D CUDA_FAST_MATH=1 \
    -D WITH_GSTREAMER=ON \
    -D WITH_OPENGL=ON \
    -D CUDA_ARCH_BIN=${ARCH_BIN} \
    -D CUDA_ARCH_PTX="" \
    -D CUDA_NVCC_FLAGS="--expt-relaxed-constexpr" \
    -D BUILD_opencv_cudacodec=ON \
    -D WITH_V4L=ON \    
    -D WITH_TBB=ON \
    -D ENABLE_FAST_MATH=1 \        
    -D WITH_QT=OFF \
    -D CMAKE_INSTALL_PREFIX=/usr/local \
    -D INSTALL_C_EXAMPLES=OFF \
    -D INSTALL_PYTHON_EXAMPLES=OFF \
    -D OPENCV_GENERATE_PKGCONFIG=ON \
    -D OPENCV_EXTRA_MODULES_PATH=~/opencv_build/opencv_contrib/modules \
    -D BUILD_EXAMPLES=OFF ..  \
    # ------------------------------
    && make -j $(nproc) \
    && make install \
    # ---------------------------------
    && rm -r ~/opencv_build \
    && rm -rf /var/lib/apt/lists/*

RUN apt-get update && apt-get install --no-install-recommends -y \
    libeigen3-dev \
    && apt autoremove && apt clean -y \
    && rm -rf /var/lib/apt/lists/*

# Compile and install tkDNN
RUN git clone https://github.com/ceccocats/tkDNN.git \
    && cd tkDNN \
    && git checkout d3372aad31d27d68593209f13e7c189752ec6e42 \
    && mkdir build \
    && cd build \
    && cmake .. \
    && make -j$(nproc)

# Run in balena
# export TKDNN_MODE=FP16
# cd /usr/src/app/tkDNN/build
# ./test_yolo4
# ./demo yolo4_fp16.rt ../demo/yolo_test.mp4 y 80 1 0

WORKDIR /usr/src/app
COPY run.sh .
CMD [ "bash", "run.sh" ]