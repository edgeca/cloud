FROM tensorflow/serving:1.14.0

# Set non-interactive for linux packages installation
ENV DEBIAN_FRONTEND=noninteractive

# Install linux packages
RUN apt-get -qq update && apt-get -qq install curl -y --no-install-recommends

# Download and extract ResNet model
RUN mkdir /tmp/resnet
RUN curl -s http://download.tensorflow.org/models/official/20181001_resnet/savedmodels/resnet_v2_fp32_savedmodel_NHWC_jpg.tar.gz | tar --strip-components=2 -C /tmp/resnet -xvz

# Copy ResNet model to /models
RUN mv /tmp/resnet /models/resnet
ENV MODEL_NAME resnet

# Remove temp and cache folders
RUN rm -rf /var/lib/apt/lists/* && rm -rf /var/cache/apt/* && rm -rf /root/.cache/* && rm -rf /install && apt-get clean