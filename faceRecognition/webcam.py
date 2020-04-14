import os
import cv2
import json
import base64
import requests
import mxnet as mx
from lib.mtcnn_detector import MtcnnDetector
import time

from datetime import datetime

detector = MtcnnDetector(model_folder='lib/models', ctx=mx.cpu(0), num_worker = 4 , accurate_landmark = False)

video_path = 0
cap = cv2.VideoCapture(video_path)

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
        Se realiza detección de rostros
        """
        img = cv2.resize(frame,(320,240))

        boxes, points = detector.detect_face(img)

        if boxes is None:
            continue

        for box in boxes:
            
            crop = frame[box[1]:box[3],box[0]:box[2],:]

            """
            Se codifica en base64 cada rostro encontrado y se envía a la API
            """
            _, enccrop = cv2.imencode('.jpg', crop)
            b64crop = base64.b64encode(enccrop)
            b64crop = b64crop.decode("utf-8")
            data = {"image": b64crop}
            headers = {'content-type': 'application/json'}

            id = post(data,headers)

            if id is not None:
                cv2.rectangle(preview,(box[0], box[1]), (box[2], box[3]),(0, 255, 0), 2)
                cv2.putText(preview, str(id["class_name"]), (box[0], box[1]-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (36,255,12), 2)
                print(id["class"])

        cv2.imshow("preview", preview)

        k = cv2.waitKey(0)
        if k == 27:    # Esc key to stop
            break
