import cv2
import glob
from FaceAlign import get_aligned_face
from FaceDetector import FaceDetectorDlib, FaceDetectorHaar
import operator
from functools import reduce
from mtcnn import MTCNN

detector = MTCNN()

def detect_and_align(path, w=250, h=300, vis=False):
    image = cv2.cvtColor(cv2.imread(path),cv2.COLOR_BGR2RGB)

    result = detector.detect_faces(image)
    print(len(result))
    if len(result) is 0:
        print(path)
        return None                             # Return None if no face is found
    else:
        box = result[0]['box']

    crop = get_aligned_face(image, box, w, h)
    graycrop = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)

    if vis:
        cv2.imshow("crop", graycrop)
        cv2.waitKey(1)

    return graycrop

# images_subjects = []
# dir = "../datasets/LookDataSet/*"
# for directory in glob.glob(dir):
#     images_subjects.append(glob.glob(directory+"/*.jpg"))
# 
# listatotal=reduce(operator.concat, images_subjects)
# for i in listatotal:
#     detect_and_align(path=i,vis=True)