
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
    # Generic tools
    build-essential \
    git \
    pkg-config \
    unzip \
    yasm \
    checkinstall \
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
# Install Tensorflow and its dependencies
# https://docs.nvidia.com/deeplearning/frameworks/install-tf-jetson-platform/index.html
# https://docs.nvidia.com/deeplearning/frameworks/install-tf-jetson-platform-release-notes/tf-jetson-rel.html#tf-jetson-rel
ENV JP_VERSION=43 \
    TF_VERSION=2.1.0
RUN apt-get update && \ 
    ldconfig && \ 
    apt-get -y install --no-install-recommends \
    libhdf5-serial-dev \
    hdf5-tools \
    libhdf5-dev \
    zlib1g-dev \
    zip \
    libjpeg8-dev \
    liblapack-dev \
    libblas-dev \
    gfortran \
    python3-h5py && \
    # Install and upgrade pip3
    apt-get -y install python3-pip && \
    pip3 install -U pip testresources \
    setuptools && \
    # Install the Python package dependencies.
    pip3 install -U numpy==1.16.1 \
    future==0.17.1 \
    mock==3.0.5 \
    gast==0.2.2 \
    futures \
    protobuf \
    pybind11 \
    # Installing TensorFlow
    && pip3 install --pre --extra-index-url \
    https://developer.download.nvidia.com/compute/redist/jp/v$JP_VERSION \
    tensorflow==$TF_VERSION \
    # ---------------------------------
    && apt autoremove && apt clean -y \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /usr/src/app
COPY run.sh .
COPY model/bisenetv2_tensorflow_depthconv.py .
CMD [ "bash", "run.sh" ]