import os
import cv2
import dlib
import numpy as np

file_folder = os.path.dirname(__file__)
model_folder = os.path.join(file_folder, "models")

# dlib models
# Frontal face detector
face_detector = dlib.get_frontal_face_detector()

# Face detector
shape_file = "shape_predictor_5_face_landmarks.dat"
shape_path = os.path.join(model_folder, shape_file)

face_shape = dlib.shape_predictor(shape_path)

# Face embedder
embedder_file = "dlib_face_recognition_resnet_model_v1.dat"
embedder_path = os.path.join(model_folder, embedder_file)

face_embedder = dlib.face_recognition_model_v1(embedder_path)

# face_detections
# Main function to get face detections in image
def face_detections(np_image):
    detections = face_detector(np_image, 1)

    return detections


# face_detections_to_box
# Changes the detections found by the model to boxes that
# are easier to handle and to plot
def face_detections_to_box(detections):
    boxes = []

    for detection in detections:
        point1_x, point1_y = detection.left(), detection.top()
        point2_x, point2_y = detection.right(), detection.bottom()

        box = [point1_x, point1_y, point2_x, point2_y]

        boxes.append(box)

    return boxes


# box_to_face_detections
# Creates the face detections objects needed by the dlib module
# in order to create the detection boxes
def boxes_to_face_detections(boxes):

    detections = dlib.rectangles()
    for box in boxes:
        point1_x, point1_y, point2_x, point2_y = box
        detection = dlib.rectangle(point1_x, point1_y, point2_x, point2_y)

        detections.append(detection)

    return detections


# align_faces creates the aligned image versions of all the detections
# found in the image
def align_faces(np_image, detections, size=150, padding=0):
    if len(detections) == 0:
        return []

    rectangles = [dlib.rectangle(det[0], det[1], det[2], det[3]) for det in detections]

    # Calculating the face points in the selected detections
    point_masks = dlib.full_object_detections()
    distances = []
    for rect in rectangles:
        # Each rect has a detection and within that face the 
        # points can be extracted and a face can be identified

        # The face_shape calculated 5 landmars in the face and those
        shape = face_shape(np_image, rect)
        point_masks.append(shape)
        
        # List of points that form the landmarks
        # points are used to align the found face
        #   0.- Right eye Right side
        #   1.- Right eye Left side
        #   2.- Left eye Left side
        #   3.- Left eye Right side
        #   4.- Tip of nose
        points = shape.parts()

        # Ratio of distances between eyes and nose
        distance_eyes = 1
        if rect.width() > 0:
            distance_eyes = (points[0].x - points[2].x) / rect.width()

        distance_nose = 0
        if rect.height() > 0:
            distance_nose = ((points[4].y - points[0].y) + (points[4].y - points[2].y)) / (2 * rect.height())

        distances.append((distance_eyes, distance_nose))

    aligned_faces = dlib.get_face_chips(np_image, point_masks, size=size, padding=padding)

    return aligned_faces, distances


# align_face Aligns an image of a detected face. 
def align_face(np_image, detection, size=150, padding=0):
    # The dlib framework uses a detection object as a dlib.rectangle
    # This rectangle for this function is the whole image since 
    # the accepted image is supposed to be only of a face

    # Calculate the 5 points in a face
    shape = face_shape(np_image, detection)

    # Adjust the face image based on the found shape
    image = dlib.get_face_chip(np_image, shape, size=size, padding=padding)

    return image


# get_faces. Returns array with all the faces found in the given image
def get_faces(np_image):
    detections = face_detections(np_image)

    # If no detection is found then None is returned
    if len(detections) == 0:
        return None

    # Looping through all the detections found in the image in order to 
    # gather the partial section in the image that corresponds to the faces
    faces = []
    for detection in detections:
        point1_x, point1_y = detection.left(), detection.top()
        point2_x, point2_y = detection.right(), detection.bottom()

        new_face = np_image[point1_y:point2_y, point1_x:point2_x]
        faces.append(new_face)

    return faces


# calculate_embedding Calculate embedding on a face found in an image
# Note that it is important to generate the aligned image as
# dlib.get_face_chip would do it i.e. the size must be 150x150, 
# centered and scaled.
def calculate_embedding(np_image, factor=1):

    # Create embedding for the face detected
    embedding = face_embedder.compute_face_descriptor(np_image)

    # L2 normalization of the calculated embedding
    l2_norm = np.linalg.norm(embedding, 2)
    embedding_norm = embedding / l2_norm * factor

    return embedding_norm
