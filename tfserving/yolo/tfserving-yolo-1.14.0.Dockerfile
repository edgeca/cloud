# Create a temp image to convert YOLO model to TF Serving SavedModel format
FROM python:3.6.9 as model_image
RUN pip3 install Keras

# Download YOLO model
RUN mkdir /tmp/yolo
COPY yolo_keras_to_tf.py /tmp/yolo

WORKDIR /tmp/yolo
RUN wget http://bit.do/ekQFf
RUN python3 --model /tmp/yolo/raccoon.h5 --target /tmp/yolo/1/

# Make Tensorflow Serving image
FROM tensorflow/serving:1.14.0 as base_image

# Set non-interactive for linux packages installation
ENV DEBIAN_FRONTEND=noninteractive

# Install linux packages
RUN apt-get -qq update && apt-get -qq install wget -y --no-install-recommends

# Copy YOLO model to /models
COPY --from=model_image /tmp/yolo/1/ /models/yolo
ENV MODEL_NAME yolo

# Remove temp and cache folders
RUN rm -rf /var/lib/apt/lists/* && rm -rf /var/cache/apt/* && rm -rf /root/.cache/* && rm -rf /install && apt-get clean