import os
import io
import cv2
import json
import base64
import requests
import numpy as np

from generator.PatchGenerator import patch_extract
from utils.Utils import green_channel

HOST = 'localhost:8502'
MODEL_NAME = 'ragnar'
VERSION = 1


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
        tampered_prob = (tampered / num_preds) * 100

        if tampered_prob >= 15:
            return "Tampered"
        else:
            return "Pristine"


def get_prediction_from_model(data):

    if data.shape != (128, 128):
        data = cv2.resize(data, (128, 128))

    payload = {"instances": [{'images': data.tolist()}]}
    r = requests.post(
        'http://' + HOST + '/v1/models/' + MODEL_NAME + ':predict',
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
