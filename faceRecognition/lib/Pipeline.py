import cv2
import logging
import numpy as np
import dlib
import os
from lib.FaceAlign import get_aligned_face
# from lib.FaceDetector import FaceDetectorDlib, FaceDetectorHaar
from mtcnn import MTCNN

logging.basicConfig(level=logging.INFO)

PUPILS_AND_NOSE = np.float32([(0.25, 0.16), (0.75, 0.16), (0.50, 0.51)])
detector = MTCNN()

# def detect_and_align(im, w=250, h=300, vis=False):
#
#     if isinstance(im, str):
#         try:
#             image = cv2.cvtColor(cv2.imread(im), cv2.COLOR_BGR2RGB)
#         except:
#             logging.error("Image type not supported")
#             return None
#
#     elif isinstance(im, np.ndarray):
#         image = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
#
#     if vis:
#         cv2.imshow("image", image)
#
#     result = detector.detect_faces(image)
#
#     if len(result) is 0:
#         print(im)
#         return None                             # Return None if no face is found
#     else:
#         box = result[0]['box']
#
#     crop = get_aligned_face(image, box, w, h)
#     graycrop = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
#
#     if vis:
#         cv2.imshow("crop", graycrop)
#         cv2.waitKey(1)
#
#     return graycrop


def detect_and_align(im, w=20, h=24, vis=False):
    if isinstance(im, str):
        try:
            original = cv2.imread(im, 1)
        except:
            logging.error("Image type not supported")
            return None
    elif isinstance(im, np.ndarray):
        original = im

    image = cv2.cvtColor(original, cv2.COLOR_BGR2RGB)
    gray = cv2.cvtColor(original, cv2.COLOR_BGR2GRAY)

    detections = detector.detect_faces(image)
    if len(detections) is 0:
        return None                             # Return None if no face is found

    box = detections[0]['box']
    landmarks = np.float32([
        detections[0]['keypoints']['left_eye'],
        detections[0]['keypoints']['right_eye'],
        detections[0]['keypoints']['nose']
    ])

    if vis:
        show = cv2.rectangle(original, (box[0], box[1]), (box[0]+box[2], box[1]+box[3]), (0, 255, 0), 2)
        show = cv2.circle(show, detections[0]['keypoints']['left_eye'], 2, (0, 255, 0), -1)
        show = cv2.circle(show, detections[0]['keypoints']['right_eye'], 2, (0, 255, 0), -1)
        show = cv2.circle(show, detections[0]['keypoints']['nose'], 2, (0, 255, 0), -1)
        cv2.imshow("box", show)
        cv2.waitKey(1)

    target_positions = PUPILS_AND_NOSE.copy()
    with np.errstate(all='ignore'):
        target_positions[:, 0] = PUPILS_AND_NOSE[:, 0]*w
        target_positions[:, 1] = PUPILS_AND_NOSE[:, 1]*h

    H = cv2.getAffineTransform(landmarks, target_positions)
    aligned = cv2.warpAffine(gray, H, (w, h))

    if vis:
        cv2.imshow("crop", aligned)
        cv2.waitKey(1)

    return aligned

# ------------------------------------------------------------------------------------------------
# file_folder = os.path.dirname(__file__)
# model_folder = os.path.join(file_folder, "models")

# # The blob size is defined by the face detector neural network
# # The model res10_300x300 being used requires a blob of 300x300
# DETECTOR_BLOB_SIZE = 300

# proto_file = "deploy.prototxt.txt"
# caffe_file = "res10_300x300_ssd_iter_140000.caffemodel"

# proto_path = os.path.join(model_folder, proto_file)
# model_path = os.path.join(model_folder, caffe_file)

# # Face detector
# face_detector = cv2.dnn.readNetFromCaffe(proto_path, model_path)

# cuda = os.environ.get("CUDA", None)
# if cuda:
#     face_detector.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
#     face_detector.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)


# # Face detector
# shape_file = "shape_predictor_5_face_landmarks.dat"
# shape_path = os.path.join(model_folder, shape_file)
# face_shape = dlib.shape_predictor(shape_path)


# # face_detection. 
# # The face detections are done using Caffe Neural Network model.
# # Only the detections above the selected threshold are sent back.
# def face_detections(np_image, face_confidence=0.8):
#     # construct a blob from the image
#     image_blob = cv2.dnn.blobFromImage(
#         np_image,
#         1.0,
#         (DETECTOR_BLOB_SIZE, DETECTOR_BLOB_SIZE),
#         (104.0, 177.0, 123.0),
#         swapRB=False,
#         crop=False
#     )

#     # apply OpenCV"s deep learning-based face detector to localize
#     # faces in the input image
#     face_detector.setInput(image_blob)

#     # Columns in detections:
#     # LowValue, HighValue, Prob, Point1_x, Point1_y, Point2_x, Point2_x
#     detections = face_detector.forward()

#     # Creating a mask with the values where the detections are larger
#     # than the confidence value for a face
#     mask = detections[:, :, :, 2] > face_confidence

#     # selected detections
#     selected_detections = detections[mask]

#     # Adjusted detections based on image size
#     # The detections coming from the neural network are based on the
#     # image blob and are in the range of 0 to 1, therefore to get
#     # the real detections the results have to be multiplied by the
#     # height and width from the image
#     (height, width) = np_image.shape[:2]

#     adjusted_detections = (selected_detections * 
#                            np.array([1, 1, 1, width, height, width, height]))

#     return adjusted_detections

# # face_detections_to_box
# # Changes the detections found by the model to boxes that
# # are easier to handle and to plot
# def face_detections_to_box(detections):
#     boxes = []

#     for detection in detections:
#         _, _, _, point1_x, point1_y, point2_x, point2_y = detection.astype("int")

#         box = [int(point1_x), int(point1_y), int(point2_x), int(point2_y)]

#         boxes.append(box)

#     return boxes

# def align_faces(np_image, detections, size=150, padding=0):
#     if len(detections) == 0:
#         return []

#     rectangles = [dlib.rectangle(det[0], det[1], det[2], det[3]) for det in detections]

#     # Calculating the face points in the selected detections
#     point_masks = dlib.full_object_detections()

#     for rect in rectangles:
#         # Each rect has a detection and within that face the 
#         # points can be extracted and a face can be identified

#         # The face_shape calculated 5 landmars in the face and those
#         shape = face_shape(np_image, rect)
#         point_masks.append(shape)
        
#         # List of points that form the landmarks
#         # points are used to align the found face
#         #   0.- Right eye Right side
#         #   1.- Right eye Left side
#         #   2.- Left eye Left side
#         #   3.- Left eye Right side
#         #   4.- Tip of nose
#         points = shape.parts()

#     aligned_faces = dlib.get_face_chips(np_image, point_masks, size=size, padding=padding)

#     return aligned_faces, points

# def detect_and_align(im, w=20, h=24, vis=False):
#     if isinstance(im, str):
#         try:
#             original = cv2.imread(im, 1)
#         except:
#             logging.error("Image type not supported")
#             return None
#     elif isinstance(im, np.ndarray):
#         original = im

#     image = cv2.cvtColor(original, cv2.COLOR_BGR2RGB)
#     gray = cv2.cvtColor(original, cv2.COLOR_BGR2GRAY)

#     result = face_detections(image)
#     boxes = face_detections_to_box(result)

#     if len(boxes) is 0:
#         return None                             # Return None if no face is found

#     box = boxes[0]

#     aligned_faces, points = align_faces(image, boxes, size=150, padding=0)
#     a_resized = cv2.resize(aligned_faces[0],(w,h),interpolation = cv2.INTER_AREA)

#     if vis:
#         show = cv2.rectangle(original, (box[0], box[1]), (box[2], box[3]), (0, 255, 0), 2)
#         show = cv2.circle(show, (points[0].x,points[0].y), 2, (0, 255, 0), -1)
#         show = cv2.circle(show, (points[1].x,points[1].y), 2, (0, 255, 0), -1)
#         show = cv2.circle(show, (points[2].x,points[2].y), 2, (0, 255, 0), -1)
#         show = cv2.circle(show, (points[3].x,points[3].y), 2, (0, 255, 0), -1)
#         show = cv2.circle(show, (points[4].x,points[4].y), 2, (0, 255, 0), -1)
#         cv2.imshow("box", show)
#         cv2.waitKey(1)

#     if vis:
#         cv2.imshow("crop", aligned_faces[0])
#         cv2.waitKey(1)

#     if vis:
#         cv2.imshow("resized", a_resized)
#         cv2.waitKey(1)

#     return a_resized
