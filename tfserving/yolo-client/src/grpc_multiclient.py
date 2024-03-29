# Copyright 2016 Google Inc. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Send JPEG image to tensorflow_model_server loaded with ResNet model.

"""

from __future__ import print_function

# This is a placeholder for a Google-internal import.

import grpc
import tensorflow as tf
import numpy as np
import cv2
from multiprocessing import Pool
# import random

from tensorflow_serving.apis import predict_pb2
from tensorflow_serving.apis import prediction_service_pb2_grpc
from tensorflow.core.framework import types_pb2
from tensorflow.contrib.util import make_tensor_proto

# The image URL is the location of the image we should send to the server
tf.app.flags.DEFINE_string('server', 'localhost:8500',
                           'PredictionService host:port')
tf.app.flags.DEFINE_string(
    'image', '/data/images/raccoon.jpg', 'path to image in JPEG format')
tf.app.flags.DEFINE_integer('batch_size', 1, 'batch size sent to tf serving')
tf.app.flags.DEFINE_integer('iterations', 1, 'iterations to run this script')
tf.app.flags.DEFINE_integer('users', 1, 'number of concurrent users')
FLAGS = tf.app.flags.FLAGS

# models = ["yolo_1", "yolo_2", "yolo_3"]


def predict(images_proto):
    options = [('grpc.max_send_message_length', 104857600),
               ('grpc.max_receive_message_length', 104857600)]
    channel = grpc.insecure_channel(FLAGS.server, options=options)
    stub = prediction_service_pb2_grpc.PredictionServiceStub(channel)
    request = predict_pb2.PredictRequest()

    model_name = "yolo"
    print("Predicting on: {}".format(model_name))

    request.model_spec.name = model_name
    request.model_spec.signature_name = "serving_default"
    request.inputs["inputs"].CopyFrom(images_proto)
    result = stub.Predict(request, 300.0)
    channel.close()
    return result


def main(_):
    net_h, net_w = 416, 416
    current_batch_size = FLAGS.batch_size

    if FLAGS.image:
        image = preprocess_input(cv2.imread(FLAGS.image), net_h, net_w)

    batch_input = np.zeros((current_batch_size, net_h, net_w, 3))
    for i in range(current_batch_size):
        batch_input[i] = image[0, :, :, :]

    images_proto = make_tensor_proto(batch_input, shape=[
                                     current_batch_size, net_h, net_w, 3], dtype=types_pb2.DT_FLOAT)

    for i in range(FLAGS.iterations):
        print("Starting iteration: {}".format(str(i)))
        with Pool(FLAGS.users) as p:
            results = p.map(
                predict, [images_proto for i in range(FLAGS.users)])
        print("Iteration {} completed.".format(str(i)))


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


if __name__ == '__main__':
    tf.app.run()
