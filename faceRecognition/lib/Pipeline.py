import cv2

from lib.FaceAlign import get_aligned_face
from lib.FaceDetector import FaceDetectorDlib, FaceDetectorHaar

detector = FaceDetectorHaar()

def detect_and_align(path, w=10, h=12, vis=False):
    image = cv2.imread(path, 1)                 # Read as BGR image

    if vis:
        cv2.imshow("image", image)
        key = cv2.waitKey(1)

    box = detector.detect(image)
    if len(box) is 0:
        return None                             # Return None if no face is found
    else:
        box = box[0]

    crop = get_aligned_face(image, box, w, h)
    graycrop = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)

    if vis:
        cv2.imshow("crop", graycrop)
        key = cv2.waitKey(1)

    return graycrop
