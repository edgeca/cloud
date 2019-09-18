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
import requests
import tensorflow as tf
import numpy as np
import cv2
from multiprocessing import Pool
import random

from tensorflow_serving.apis import predict_pb2
from tensorflow_serving.apis import prediction_service_pb2_grpc
from tensorflow.core.framework import types_pb2
from tensorflow.contrib.util import make_tensor_proto

tf.app.flags.DEFINE_string('url', 'http://yolo-api-service/predict',
                           'URL for prediction')
tf.app.flags.DEFINE_integer('batch_size', 1, 'batch size sent to tf serving')
tf.app.flags.DEFINE_integer('iterations', 1, 'iterations to run this script')
tf.app.flags.DEFINE_integer('users', 1, 'number of concurrent users')
FLAGS = tf.app.flags.FLAGS

# models = ["yolo_1", "yolo_2", "yolo_3"]


def predict(batch_size):
    # model_name = random.choice(models)
    model_name = "yolo"
    print("Predicting on: {}".format(model_name))
    r = requests.get(FLAGS.url, params={
        'model_name': model_name,
        'batch_size': batch_size
    })
    return r.text


def main(_):
    batch_size = FLAGS.batch_size

    for i in range(FLAGS.iterations):
        print("Starting iteration: {}".format(str(i)))
        with Pool(FLAGS.users) as p:
            results = p.map(
                predict, [batch_size for i in range(FLAGS.users)])
        print("Iteration {} completed.".format(str(i)))


if __name__ == '__main__':
    tf.app.run()
