import os
import json
import argparse
from flask import Flask, Response, request
from timeit import default_timer as timer

import grpc
import tensorflow as tf
import numpy as np
import cv2

from tensorflow_serving.apis import predict_pb2
from tensorflow_serving.apis import prediction_service_pb2_grpc
from tensorflow.core.framework import types_pb2
from tensorflow.contrib.util import make_tensor_proto

app = Flask(__name__)

server = "yolo-service:8500"
image_path = "/data/images/raccoon.jpg"


def preprocess_input(image, net_h, net_w):
    new_h, new_w, _ = image.shape

    # determine the new size of the image
    if (float(net_w)/new_w) < (float(net_h)/new_h):
        new_h = (new_h * net_w)//new_w
        new_w = net_w
    else:
        new_w = (new_w * net_h)//new_h
        new_h = net_h

    # resize the image to the new size
    resized = cv2.resize(image[:, :, ::-1]/255., (new_w, new_h))

    # embed the image into the standard letter box
    new_image = np.ones((net_h, net_w, 3)) * 0.5
    new_image[(net_h-new_h)//2:(net_h+new_h)//2,
              (net_w-new_w)//2:(net_w+new_w)//2, :] = resized
    new_image = np.expand_dims(new_image, 0)

    return new_image


def predict(model_name, images_proto):
    options = [('grpc.max_send_message_length', 104857600),
               ('grpc.max_receive_message_length', 104857600)]
    channel = grpc.insecure_channel(server, options=options)
    stub = prediction_service_pb2_grpc.PredictionServiceStub(channel)
    request = predict_pb2.PredictRequest()
    request.model_spec.name = model_name
    request.model_spec.signature_name = "serving_default"
    request.inputs["inputs"].CopyFrom(images_proto)
    result = stub.Predict(request, 300.0)
    channel.close()
    return result


@app.route("/predict")
def predict_api():

    model_name = request.args.get("model_name", default="yolo_1", type=str)
    batch_size = request.args.get("batch_size", default=1, type=int)

    net_h, net_w = 416, 416
    image = preprocess_input(cv2.imread(image_path), net_h, net_w)

    batch_input = np.zeros((batch_size, net_h, net_w, 3))
    for i in range(batch_size):
        batch_input[i] = image[0, :, :, :]

    images_proto = make_tensor_proto(batch_input, shape=[
                                     batch_size, net_h, net_w, 3], dtype=types_pb2.DT_FLOAT)

    start_pred = timer()
    predict(model_name, images_proto)
    end_pred = timer()

    print("Completed pred for images(s) in {:0.2f}s".format(
        end_pred - start_pred))

    return "success"


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--server', help='Prediction Service host:port', default='localhost:8500', type=str)
    parser.add_argument(
        '--image', help='path to image in JPEG format', default='/data/images/raccoon.jpg', type=str)
    args = parser.parse_args()
    app.run(host="0.0.0.0", port=80)
