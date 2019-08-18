"""Send JPEG image to tensorflow_model_server loaded with YOLOv3 raccoon model.

"""

from __future__ import print_function

import grpc
import tensorflow as tf
import numpy as np
import cv2

from tensorflow_serving.apis import predict_pb2
from tensorflow_serving.apis import prediction_service_pb2_grpc
from tensorflow.core.framework import types_pb2
from tensorflow.contrib.util import make_tensor_proto

tf.app.flags.DEFINE_string('server', 'localhost:8500',
                           'PredictionService host:port')
tf.app.flags.DEFINE_string('image', '/data/images/raccoon.jpg', 'path to image in JPEG format')
tf.app.flags.DEFINE_integer('batch_size', 1, 'batch size sent to tf serving')
tf.app.flags.DEFINE_integer('iterations', 1, 'iterations to run this script')
FLAGS = tf.app.flags.FLAGS


def main(_):
    net_h, net_w = 416, 416
    current_batch_size = FLAGS.iterations
    
    if FLAGS.image:
        image = preprocess_input(cv2.imread(FLAGS.image), net_h, net_w)
    
    batch_input = [image for i in range(current_batch_size)]
    images_proto = make_tensor_proto(batch_input, shape=[current_batch_size, net_h, net_w, 3], dtype=types_pb2.DT_FLOAT)
    options = [('grpc.max_send_message_length', 104857600), ('grpc.max_receive_message_length', 104857600)]

    for i in range(FLAGS.iterations):
        print("Starting iteration: {}".format(str(i)))

        try:
            channel = grpc.insecure_channel(FLAGS.server, options=options)
            stub = prediction_service_pb2_grpc.PredictionServiceStub(channel)
            request = predict_pb2.PredictRequest()
            request.model_spec.name = "yolo"
            request.model_spec.signature_name = "serving_default"
            request.inputs[input_feature].CopyFrom(images_proto)
            result = stub.Predict(request, 120.0)
        except Exception as ex:
            print("Error in predictions on model {}".format(str(ex)))
        finally:
            channel.close()

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

def predict_on_images(model_name, signature_name, input_feature, images):
        """Runs prediction on images

        Runs prediction on given images inside Tensorflow Serving container for given model.

        Parameters
        ----------
        model_name - string
            Name of the table detection model
        signature_name - string
            Model signature
        input_feature - string
            Name of the input feature
        images - TensorProto
            List of images as TensorProto

        Returns
        -------
        dict
            Result as dictionary
        """
        try:
            channel = self.get_channel()
            stub = prediction_service_pb2_grpc.PredictionServiceStub(channel)
            request = predict_pb2.PredictRequest()
            request.model_spec.name = model_name
            request.model_spec.signature_name = signature_name
            request.inputs[input_feature].CopyFrom(images)
            result = stub.Predict(request, self.timeout)
            return result
        except Exception as ex:
            logger.error(
                "Error in predictions on model {} - {}".format(model_name, str(ex)))
            return None
        finally:
            channel.close()

if __name__ == '__main__':
    tf.app.run()
