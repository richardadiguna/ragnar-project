import os
import io
import cv2
import json
import base64
import requests
import numpy as np

from bunch import Bunch
from flask import Flask
from flask import request
from flask import jsonify
from generator.PatchGenerator import tf_patch_extract
from utils.Utils import green_channel

UPLOAD_FOLDER = '/temp'
ALLOWED_EXTENSIONS = set(['jpg', 'jpeg'])

HOST = 'localhost:8501'
MODEL_NAME = 'ragnar'
VERSION = 1

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER


def prediction_summary(preds):
    pristine = 0
    tampered = 0
    num_preds = preds.shape[0]
    class_predictions = np.argmax(preds, axis=1)

    for out in class_predictions:
        if out == 1 or out == 2 or \
           out == 3 or out == 4 or \
           out == 5:
            tampered += 1
        else:
            pristine += 1

    if tampered >= pristine:
        return 'Tampered'
    else:
        tampered_prob = (tampered/num_preds)*100

        if tampered_prob >= 15:
            return "Tampered"
        else:
            return "Pristine"


def get_prediction_from_model(data):

    if data.shape != (128, 128):
        data = cv2.resize(data, (128, 128))

    payload = {"instances": [{'images': data.tolist()}]}
    r = requests.post(
        'http://localhost:8501/v1/models/kratos:predict',
        json=payload)
    b = r.content.decode('utf8').replace("'", '"')
    c_data = json.loads(b)
    response = Bunch(c_data)

    rank = np.argmax(response.predictions)

    if rank == 0:
        return 'background'
    elif rank == 1:
        return 'e-ktp'
    elif rank == 2:
        return 'ktp'
    elif rank == 3:
        return 'dukcapil'

    return 'undefined'


def convert_im_file(file):
    in_memory_file = io.BytesIO()
    file.save(in_memory_file)
    data = np.fromstring(
        in_memory_file.getvalue(),
        dtype=np.uint8)
    color_image_flag = 1
    return cv2.imdecode(data, color_image_flag)


def allowed_file(filename):
    return '.' in filename and \
            filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


@app.route("/prediction", methods=['POST'])
def get_prediction():
    if request.method == 'POST':

        if 'image' not in request.files:
            print('No file part, key: image')
            return jsonify({"error": "No file part, key: image"})

        file = request.files['image']

        if file.filename == '':
            print('No selected file')
            return jsonify({"error": "No selected file"})

        if file and allowed_file(file.filename):
            img_decoded = convert_im_file(file)
            result = get_prediction_from_model(img_decoded)
            return jsonify({"result": result})

    return jsonify({"error": "500"})


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=3000, debug=True)
