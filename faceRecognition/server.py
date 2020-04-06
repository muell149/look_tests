import cv2
import base64
import logging
import argparse
import numpy as np

from flask import Flask, Response, request, jsonify
from functions2 import DataSet

logging.basicConfig(level=logging.INFO)

app = Flask(
    __name__
)


@app.route("/", methods=["POST"])
def index():
    data = request.get_json()

    src = data["image"]
    img_decoded = base64.standard_b64decode(src)
    img_buffer = np.frombuffer(img_decoded, np.uint8)
    img = cv2.imdecode(img_buffer, 1)

    res = DS.classify(img)

    return jsonify({
        "class": res
    })


@app.route("/info", methods=["GET"])
def info():
    return json.dumps({
        "number_classes": DS.number_classes
    })


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--directory", "-dir", help="Dataset directory", type=str, default="datasets/lfw/*")
    parser.add_argument("--extension", "-ext", help="Dataset images extension", type=str, default="jpg")
    parser.add_argument("--images_per_class", "-ipc", help="Images to use per class", type=int, default=20)
    parser.add_argument("--width", "-wi", help="Image width", type=int, default=10)
    parser.add_argument("--height", "-he", help="Image height", type=int, default=12)
    parser.add_argument("--vertical", "-ve", help="Vertical splits", type=int, default=4)
    parser.add_argument("--horizontal", "-ho", help="Horizontal splits", type=int, default=2)
    parser.add_argument("--epsilon", "-e", help="Epsilon", type=float, default=0.0)
    parser.add_argument("--threshold", "-t", help="Classification threshold", type=float, default=0.22)
    args = parser.parse_args()

    DS = DataSet(
        dir=args.directory,
        ext=args.extension,
        images_per_class=args.images_per_class,
        width=args.width,
        height=args.height,
        vertical=args.vertical,
        horizontal=args.horizontal,
        epsilon=args.epsilon,
        threshold=args.threshold,
        vis=False
    )

    app.run(
        host="0.0.0.0",
        port="8000",
        threaded=True,
        debug=True,
        use_reloader=False
    )