import cv2
import dlib
from numpy import clip

swaprb = lambda x: x[...,::-1]

class FaceDetectorDlib:
    """
    Detector de rostros de la librería Dlib. Es el detector más preciso pero
    el más pesado.
    """
    def __init__(self):
        self.detector = dlib.get_frontal_face_detector()

    def detect(self, image):
        image = swaprb(image)
        dets = self.detector(image, 1)
        return [((d.left(), d.top()), (d.right(), d.bottom())) for d in dets]


class FaceDetectorHaar:
    """
    Detector de rostros con HaarCascade. Es el más rápido pero el menos preciso.
    Se amplía la detección por un factor "amp" para mejorar el espacio de
    detección.
    """
    def __init__(self, amp=0.35):
        self.face_cascade = cv2.CascadeClassifier('models/haarcascade_frontalface_default.xml')
        self.amp = amp

    def detect(self, image):
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        imh, imw = gray.shape
        faces = self.face_cascade.detectMultiScale(gray, 1.3, 5)

        clips = [(clip(x-int(w*self.amp),0,imw), clip(y-int(h*self.amp),0,imh), clip(x+w+int(w*self.amp),0,imw), clip(y+h+int(h*self.amp),0,imh)) for (x,y,w,h) in faces]
        return [((l,t), (r,b)) for (l,t,r,b) in clips]
        # return [(((l,t), (r,b)), (slice(t,b), slice(l,r))) for (l,t,r,b) in clips]


class FaceDetectorSSD:
    def __init__(self):
        pass

    def detect(self, image):
        pass
