import os
import cv2
import json
import base64
import requests

from datetime import datetime
from mtcnn import MTCNN
from lib.FaceDetectorHaar import FaceDetectorHaar

video_path = 0
cap = cv2.VideoCapture(video_path)

face_detector = "MTCNN"



if face_detector == "MTCNN":
    detector = MTCNN()
elif face_detector == "HAAR":
    detector = FaceDetectorHaar()

def post(data, headers):
    """
    Función para realizar peticiones a la API de reconocimiento facial
    """
    try:
        r = requests.post("http://localhost:8000/", data=json.dumps(data), headers=headers)
        resp = json.loads(r.text)
        return resp
    except ConnectionRefusedError:
        return None


while True:
    ok, frame = cap.read()
    if ok:
        preview = frame.copy()
        """
        Se realiza detección de rostros con HaarCascade sobre el frame
        """
        if face_detector == "MTCNN":

            faces = detector.detect_faces(frame)

            for face in faces:
                box = face["box"]
                
                crop = frame[box[1]:box[1]+box[3],box[0]:box[0]+box[2],:]

                """
                Se codifica en base64 cada rostro encontrado y se envía a la API
                """
                _, enccrop = cv2.imencode('.jpg', crop)
                b64crop = base64.b64encode(enccrop)
                b64crop = b64crop.decode("utf-8")
                data = {"image": b64crop}
                headers = {'content-type': 'application/json'}

                id = post(data, headers)
                if id is not None:
                    cv2.rectangle(preview,(box[0], box[1]), (box[0]+box[2], box[1]+box[3]),(0, 255, 0), 2)
                    cv2.putText(preview, str(id["class_name"]), (box[0], box[1]-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (36,255,12), 2)
                    print(id["class"])

            cv2.imshow("preview", preview)

            k = cv2.waitKey(0)
            if k == 27:    # Esc key to stop
                break

        elif face_detector == "HAAR":

            boxes = detector.detect(frame)

            for box, (sy, sx) in boxes:
                print(box)
                cv2.rectangle(preview, box[0], box[1], (0, 255, 0), 1)
                crop = frame[sy, sx, :]

                """
                Se codifica en base64 cada rostro encontrado y se envía a la API
                """
                _, enccrop = cv2.imencode('.jpg', crop)
                b64crop = base64.b64encode(enccrop)
                b64crop = b64crop.decode("utf-8")
                data = {"image": b64crop}
                headers = {'content-type': 'application/json'}

                id = post(data, headers)
                if id is not None:
                    print(id)

            cv2.imshow("preview", preview)

            k = cv2.waitKey(0)
            if k == 27:    # Esc key to stop
                break