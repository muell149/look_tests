import os
import cv2

import face_detection.dlib_models as dlib_models

dir_path = os.path.dirname(__file__)

def test_one_face():
    image_path = os.path.join(dir_path, "pictures/face.jpg")
    np_image = cv2.imread(image_path)

    detections = dlib_models.face_detections(np_image)

    assert len(detections) == 1


def test_group_face():
    image_path = os.path.join(dir_path, "pictures/group.jpg")
    np_image = cv2.imread(image_path)

    detections = dlib_models.face_detections(np_image)

    assert len(detections) == 6


def test_no_face():
    image_path = os.path.join(dir_path, "pictures/noface.jpg")
    np_image = cv2.imread(image_path)

    detections = dlib_models.face_detections(np_image)

    assert len(detections) == 0