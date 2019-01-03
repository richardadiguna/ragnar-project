from __future__ import print_function

import os
import io
import cv2
import json
import base64
import requests
import numpy as np
import tensorflow as tf

from grpc.beta import implementation
from tensorflow_serving.apis import predict_pb2
from tensorflow_serving.apis import prediction_service_pb2_grpc
from bunch import Bunch
from generator.PatchGenerator import patch_extract
from utils.Utils import green_channel

HOST = '130.211.244.102:8502'
MODEL_NAME = 'ragnar'
VERSION = 1

ALLOWED_EXTENSIONS = set(['jpg', 'jpeg'])


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

    data = green_channel(data)
    patches, _, _ = patch_extract(data, 128, 64)

    channel = implementations.insecure_channel('localhost', 8500)
    stub = prediction_service_pb2_grpc.cre(channel)

    request = predict_pb2.PredictRequest()
    request.model_spec.name = 'ragnar'
    request.model_spec.signature_name = ''

    request.inputs['images'].CopyFrom(
        tf.contrib.util.make_tensor_proto(
            patches[0], dtype=float32, shape=[1, 128, 128, 1]))
    request.inputs['trainable'].CopyFrom(
        tf.contrib.util.make_tensor_proto(
            False, dtype=bool))

    result = stub.Predict(request, 10.0)

    c_data = json.loads(result)
    response = Bunch(c_data)

    result = prediction_summary(response.predictions)

    return result


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
