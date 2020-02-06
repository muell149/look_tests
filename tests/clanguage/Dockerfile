FROM ubuntu:18.04

RUN apt-get update -y
RUN apt-get upgrade -y
RUN apt-get update -y

#**************** Install C++ and OpenCV *****************
RUN apt-get install -y build-essential cmake git pkg-config libgtk-3-dev libavcodec-dev \
    libavformat-dev libswscale-dev libv4l-dev libxvidcore-dev \
    libx264-dev libjpeg-dev libpng-dev libtiff-dev gfortran openexr libatlas-base-dev python3-dev \
    python3-numpy libtbb2 libtbb-dev libdc1394-22-dev unzip nano
RUN mkdir ~/opencv_build
WORKDIR /root/opencv_build
RUN git clone --verbose https://github.com/opencv/opencv.git
RUN git clone --verbose https://github.com/opencv/opencv_contrib.git
WORKDIR /root/opencv_build/opencv
RUN mkdir build
WORKDIR /root/opencv_build/opencv/build
RUN cmake -D CMAKE_BUILD_TYPE=RELEASE \
    -D CMAKE_INSTALL_PREFIX=/usr/local \
    -D INSTALL_C_EXAMPLES=ON \
    -D INSTALL_PYTHON_EXAMPLES=ON \
    -D OPENCV_EXTRA_MODULES_PATH=~/opencv_build/opencv_contrib/modules \
    -D BUILD_EXAMPLES=ON ..
RUN make -j1
RUN make install

WORKDIR /app

RUN wget https://download.pytorch.org/libtorch/cpu/libtorch-cxx11-abi-shared-with-deps-1.4.0%2Bcpu.zip
RUN unzip *.zip

ADD . /app

