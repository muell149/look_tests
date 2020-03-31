FROM nvidia/cuda:9.2-runtime-ubuntu18.04

WORKDIR /app

ADD . /app

RUN apt-get -y update 
RUN apt-get -y upgrade 
RUN apt-get -y update 

RUN apt-get -y install python3-pip
RUN pip3 install --upgrade pip

RUN pip3 install -r requirements.txt