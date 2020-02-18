FROM ubuntu:18.04

RUN apt-get update -y
RUN apt-get upgrade -y
RUN apt-get install libatlas-base-dev gfortran -y
RUN apt-get install python3-dev -y
RUN apt-get install python3-pip -y
RUN apt-get install python3-numpy python3-scipy -y
RUN pip3 install cvxpy
RUN pip3 install matplotlib
RUN pip3 install -U scikit-learn

ADD lfw.py /
RUN python3 /lfw.py
RUN mkdir app

ADD facialRecognition.py /app
ADD functions.py /app

WORKDIR /app

