from __future__ import print_function

import os
import io
import cv2
import json
import base64
import requests
import numpy as np
import tensorflow as tf

from grpc.beta import implementations
from tensorflow_serving.apis import predict_pb2
from tensorflow_serving.apis import prediction_service_pb2
from bunch import Bunch
from generator.PatchGenerator import patch_extract
from utils.Utils import green_channel


HOST = '35.194.178.163:8500'
MODEL_NAME = 'ragnar'
VERSION = 1

ALLOWED_EXTENSIONS = set(['jpg', 'jpeg'])
NUM_CLASSES = 6


def prediction_summary(scores, threshold):
    n = len(scores)
    count = 0

    preds = np.zeros(
        shape=((n // NUM_CLASSES), NUM_CLASSES),
        dtype=np.float32)

    for i in range(n):

        if i == 0:
            continue

        if i % NUM_CLASSES == 0:
            probs = scores[i-NUM_CLASSES:i]
            probs_arr = np.array(probs, dtype=np.float32)
            preds[count] = probs_arr
            count += 1

    predictions = np.argmax(preds, axis=1)
    unique, counts = np.unique(predictions, return_counts=True)
    summary = dict(zip(unique, counts))

    # summary with key 0 has value the numbers of pristine patches
    # keys besides 0 has value the numbers of tampered patches
    pristine = 0

    # Pristine class
    if summary[0]:
        pristine += summary[0]

    # JPEG compression class
    # exclude this for being a tampered class
    if summary[5]:
        pristine += summary[5]

    pristine_pa = (pristine / (n // NUM_CLASSES)) * 100.0

    if pristine_pa < threshold:
        return 'Tampered'

    return 'Pristine'


def get_prediction_from_model(data):
    host = HOST.split(':')
    data = green_channel(data)
    patches, _, _ = patch_extract(data, 128, 64)
    channel = implementations.insecure_channel(host[0], int(host[1]))
    stub = prediction_service_pb2.beta_create_PredictionService_stub(channel)

    request = predict_pb2.PredictRequest()
    request.model_spec.name = 'ragnar'
    request.model_spec.signature_name = ''

    request.inputs['images'].CopyFrom(
        tf.contrib.util.make_tensor_proto(
            patches, dtype=np.float32,
            shape=[patches.shape[0], 128, 128, 1]))
    request.inputs['trainable'].CopyFrom(
        tf.contrib.util.make_tensor_proto(
            False, dtype=bool))

    result = stub.Predict(request, 100.0)
    scores = np.array(result.outputs['scores'].float_val)

    out = prediction_summary(scores.tolist(), 85.0)

    return out


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
