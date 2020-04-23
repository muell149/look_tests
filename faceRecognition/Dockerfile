#FROM ubuntu:18.04
FROM nvidia/cuda:10.2-runtime-ubuntu18.04

RUN apt-get update -y
RUN apt-get upgrade -y
RUN apt-get install -y gcc g++ clang make gdb valgrind graphviz imagemagick zip unzip python3-pip build-essential cmake libboost-all-dev python3-dev libsm6 libxext6 libxrender-dev libglib2.0-0
RUN apt-get update -y
RUN apt-get upgrade -y
RUN pip3 install --upgrade pip
ADD requirements.txt requirements.txt
RUN pip install -r requirements.txt

WORKDIR /app
COPY . .

CMD ["python3","testing_accuracy.py"]