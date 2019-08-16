FROM tensorflow/serving:1.14.0

# Download and extract ResNet model
RUN mkdir /tmp/resnet
RUN curl -s http://download.tensorflow.org/models/official/20181001_resnet/savedmodels/resnet_v2_fp32_savedmodel_NHWC_jpg.tar.gz | \
RUN tar --strip-components=2 -C /tmp/resnet -xvz

# Copy ResNet model to /models
RUN mv /tmp/resnet /models/resnet
ENV MODEL_NAME resnet