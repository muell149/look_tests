import os
import cv2

import face_detection.dlib_models as dlib_models
import face_detection.openface_models as openface_models

dir_path = os.path.dirname(__file__)

FACTOR = 10000
JITTERS = 5

def create_embedding():
    image_path = os.path.join(dir_path, "pictures/face.jpg")
    np_image = cv2.imread(image_path)

    # Face detections
    detections = dlib_models.face_detections(np_image)

    # Face alignment
    np_align = dlib_models.align_face(np_image, detections[0])

    print("--------------------")
    print("dlib embedding example")
    embedding_dlib = dlib_models.calculate_embedding(np_align, FACTOR)
    print("Type: ", type(embedding_dlib))
    print(embedding_dlib)

    # Create transformations with both methods
    print("--------------------")
    print("dlib transformations")
    embeddings_dlib = dlib_models.calculate_transformations(
        np_align, 
        embedder=dlib_models.calculate_embedding,
        embedding_factor=FACTOR,
        num_jitters=JITTERS,
    )
    for emb in embeddings_dlib:
        print("*")
        print(emb)
        print("*")

    print("Number of embeddings: ", len(embeddings_dlib))
    print("Size of embedding: ", len(embeddings_dlib[0].split(",")))
    
    ##################################
    # Embeddings calculated with NN
    # Create embedding with aligned image
    print("\n--------------------")
    print("NN embedding example")
    embedding_nn = openface_models.calculate_embedding(np_align, FACTOR)
    print("Type: ", type(embedding_nn))
    print(embedding_nn)

    print("--------------------")
    print("NN transformations")
    embeddings_nn = dlib_models.calculate_transformations(
        np_align, 
        embedder=openface_models.calculate_embedding,
        embedding_factor=FACTOR,
        num_jitters=JITTERS,
    )
    for emb in embeddings_nn:
        print("*")
        print(emb)
        print("*")

    print("Number of embeddings: ", len(embeddings_nn))
    print("Size of embedding: ", len(embeddings_nn[0].split(",")))


if __name__ == "__main__":
    create_embedding()