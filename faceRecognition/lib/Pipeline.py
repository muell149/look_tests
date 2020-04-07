import cv2
import logging
import numpy as np

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
