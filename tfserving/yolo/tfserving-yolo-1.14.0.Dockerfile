# Create a temp image to convert YOLO model to TF Serving SavedModel format
FROM python:3.6.9 as model_image
RUN pip3 install Keras==2.2.4 tensorflow==1.12.0

# Download YOLO model
RUN mkdir /tmp/yolo
COPY yolo_keras_to_tf.py /tmp/yolo

WORKDIR /tmp/yolo
RUN wget -O raccoon.h5 -nv --no-check-certificate "https://onedrive.live.com/download?cid=5FDEBAB7450CDD92&resid=5FDEBAB7450CDD92%21137&authkey=AB5oeSO0Kr5yfTo"
RUN python3 yolo_keras_to_tf.py --model=/tmp/yolo/raccoon.h5 --target=/tmp/yolo/1/

# Make Tensorflow Serving image
FROM tensorflow/serving:1.14.0 as base_image

# Set non-interactive for linux packages installation
ENV DEBIAN_FRONTEND=noninteractive

# Install linux packages
RUN apt-get -qq update && apt-get -qq install wget -y --no-install-recommends

# Copy YOLO model to /models
COPY --from=model_image /tmp/yolo/1 /models/yolo/1
COPY models.config /models/models.config

# Copy monitoring script
COPY monitor_serving.sh /usr/bin/monitor_serving.sh
RUN chmod a+x /usr/bin/monitor_serving.sh

# Remove temp and cache folders
RUN rm -rf /var/lib/apt/lists/* && rm -rf /var/cache/apt/* && rm -rf /root/.cache/* && rm -rf /install && apt-get clean

ENTRYPOINT []