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

from tensorflow_serving.apis import predict_pb2
from tensorflow_serving.apis import prediction_service_pb2_grpc
from multiprocessing import Pool

# The image URL is the location of the image we should send to the server
IMAGE_URL = 'https://tensorflow.org/images/blogs/serving/cat.jpg'

tf.app.flags.DEFINE_string('server', 'localhost:8500',
                           'PredictionService host:port')
tf.app.flags.DEFINE_string('image', '', 'path to image in JPEG format')
tf.app.flags.DEFINE_integer('batch_size', 1, 'batch size sent to tf serving')
tf.app.flags.DEFINE_integer('iterations', 1, 'iterations to run this script')
tf.app.flags.DEFINE_integer('users', 1, 'number of concurrent users')
FLAGS = tf.app.flags.FLAGS


def predict(data):
    channel = grpc.insecure_channel(FLAGS.server)
    stub = prediction_service_pb2_grpc.PredictionServiceStub(channel)
    request = predict_pb2.PredictRequest()
    request.model_spec.name = 'resnet'
    request.model_spec.signature_name = 'serving_default'
    request.inputs['image_bytes'].CopyFrom(
        tf.contrib.util.make_tensor_proto(data, shape=[FLAGS.batch_size]))
    result = stub.Predict(request, 60.0)
    return result


def main(_):
    if FLAGS.image:
        with open(FLAGS.image, 'rb') as f:
            data = f.read()
    else:
        # Download the image since we weren't given one
        dl_request = requests.get(IMAGE_URL, stream=True)
        dl_request.raise_for_status()
        data = dl_request.content

    data = [data for i in range(FLAGS.batch_size)]

    for i in range(FLAGS.iterations):
        print("Starting iteration: {}".format(str(i)))
        with Pool(FLAGS.users) as p:
            results = p.starmap(predict, [data for i in range(FLAGS.users)])
        print("Iteration {} completed.".format(str(i)))


if __name__ == '__main__':
    tf.app.run()
