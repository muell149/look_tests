import dlib

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
