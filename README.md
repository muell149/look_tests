# Look Tests

## Setup:


## Introduction 
This is a full face detection and recognition system. For the detection part [MTCNN](https://github.com/ipazc/mtcnn) package is used. The recognition is achieved by using both a sparse representation method and using a Keras implementation of [FaceNet](https://github.com/faustomorales/keras-facenet) to get the embeddings.

## Getting Started
### Embeddings
1.	Installation process
    
    You can simply clone this repository. After that, you need to unzip the simple dataset contained in face_recognition_embeddings/datasets.

2.	Software dependencies
    
    You can see the necessary pip packages on the file face_recognition_embeddings/requirements.txt. Also, install cmake on your system, in order to install it on a debian-based distribution run
    
        sudo apt-get install -y cmake

3.  Inside face_recognition_embeddings directory, run the command

        python3 main.py
