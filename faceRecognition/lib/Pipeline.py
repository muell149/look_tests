import cv2
from lib.FaceAlign import get_aligned_face
from lib.FaceDetector import FaceDetectorDlib, FaceDetectorHaar
from mtcnn import MTCNN
import numpy as np

logging.basicConfig(level=logging.INFO)

detector = MTCNN()

def detect_and_align(im, w=250, h=300, vis=False):

    if isinstance(im, str):
        try:
            image = cv2.cvtColor(cv2.imread(im), cv2.COLOR_BGR2RGB)
        except:
            logging.error("Image type not supported")
            return None

    elif isinstance(im, np.darray):
        image = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)

    if vis:
        cv2.imshow("image", image)

    result = detector.detect_faces(image)

    if len(result) is 0:
        print(im)
        return None                             # Return None if no face is found
    else:
        box = result[0]['box']

    crop = get_aligned_face(image, box, w, h)
    graycrop = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)

    if vis:
        cv2.imshow("crop", graycrop)
        cv2.waitKey(1)

    return graycrop