import os
import cv2
import numpy as np
import imutils

file_folder = os.path.dirname(__file__)
model_folder = os.path.join(file_folder, "models")

# The blob size is defined by the face detector neural network
# The model res10_300x300 being used requires a blob of 300x300
DETECTOR_BLOB_SIZE = 300

proto_file = "deploy.prototxt.txt"
caffe_file = "res10_300x300_ssd_iter_140000.caffemodel"

proto_path = os.path.join(model_folder, proto_file)
model_path = os.path.join(model_folder, caffe_file)

# Face detector
face_detector = cv2.dnn.readNetFromCaffe(proto_path, model_path)

cuda = os.environ.get("CUDA", None)
if cuda:
    face_detector.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
    face_detector.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)

# The blob size is defined by the embedding calculator neural network
# The model nn4.small2.v1.t7 being used requires a blob of 96x96
EMBEDDING_BLOB_SIZE = 80

embedding_file = "nn4.small2.v1.t7"
embedding_model = os.path.join(model_folder, embedding_file)

# Face embedder
face_embedder = cv2.dnn.readNetFromTorch(embedding_model)
if cuda:
    face_embedder.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
    face_embedder.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)


# face_detection. 
# The face detections are done using Caffe Neural Network model.
# Only the detections above the selected threshold are sent back.
def face_detections(np_image, face_confidence=0.8):

    # construct a blob from the image
    image_blob = cv2.dnn.blobFromImage(
        np_image,
        1.0,
        (DETECTOR_BLOB_SIZE, DETECTOR_BLOB_SIZE),
        (104.0, 177.0, 123.0),
        swapRB=False,
        crop=False
    )

    # apply OpenCV"s deep learning-based face detector to localize
    # faces in the input image
    face_detector.setInput(image_blob)

    # Columns in detections:
    # LowValue, HighValue, Prob, Point1_x, Point1_y, Point2_x, Point2_x
    detections = face_detector.forward()

    # Creating a mask with the values where the detections are larger
    # than the confidence value for a face
    mask = detections[:, :, :, 2] > face_confidence

    # selected detections
    selected_detections = detections[mask]

    # Adjusted detections based on image size
    # The detections coming from the neural network are based on the
    # image blob and are in the range of 0 to 1, therefore to get
    # the real detections the results have to be multiplied by the
    # height and width from the image
    (height, width) = np_image.shape[:2]

    adjusted_detections = (selected_detections * 
                           np.array([1, 1, 1, width, height, width, height]))

    return adjusted_detections


# face_detections_to_box
# Changes the detections found by the model to boxes that
# are easier to handle and to plot
def face_detections_to_box(detections):
    boxes = []

    for detection in detections:
        _, _, _, point1_x, point1_y, point2_x, point2_y = detection.astype("int")

        box = [int(point1_x), int(point1_y), int(point2_x), int(point2_y)]

        boxes.append(box)

    return boxes


# detect_face_in_image
# Returns the detection with the highest probability
def detect_face_in_image(np_image, face_confidence=0.8):

    # Calculating all face detections above confidence threshold
    detections = face_detections(np_image, face_confidence)

    # If there are more than one face, the detection with the hightest
    # probability is selected and returned
    # The detection array is sorted by the probability column
    detection = []
    if len(detections) >= 1:
        sorted_rows = detections[:,2].argsort()[::-1]
        sorted_detections = detections[sorted_rows, :]

        detection = [sorted_detections[0, :]]

    return detection


# get_faces. Returns array with all the faces found in the given image
def get_faces(np_image, face_confidence=0.8):
    detections = face_detections(np_image, face_confidence)

    # If no detection is found then None is returned
    if len(detections) == 0:
        return None

    # Looping through all the detections found in the image in order to 
    # gather the partial section in the image that corresponds to the faces
    faces = []
    for detection in detections:
        _, _, _, point1_x, point1_y, point2_x, point2_y = detection.astype("int")

        new_face = np_image[point1_y:point2_y, point1_x:point2_x]
        faces.append(new_face)

    return faces


# calculate_embedding. Creates embedding for the found face
# To calculate the embedding an image of a face is sent to the function
def calculate_embedding(np_image, factor=1, blob_size=EMBEDDING_BLOB_SIZE):

    face_blob = cv2.dnn.blobFromImage(
        np_image,
        1.0 / 255,
        (blob_size, blob_size),
        (0, 0, 0),
        swapRB=True,
        crop=False)

    # Setting the image to be converted to embeddings
    face_embedder.setInput(face_blob)

    # Calculating the embeddings for the found face
    embedding = face_embedder.forward()[0]

    # L2 normalization of the calculated embedding
    l2_norm = np.linalg.norm(embedding, 2)
    embedding_norm = embedding / l2_norm * factor

    return embedding_norm
