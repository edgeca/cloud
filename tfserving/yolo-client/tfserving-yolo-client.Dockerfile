FROM python:3.6.9

# Set non-interactive for linux packages installation
ENV DEBIAN_FRONTEND=noninteractive

# Install linux packages
RUN apt-get -qq update && apt-get -qq install curl -y --no-install-recommends
RUN apt-get -qq update && apt-get -qq install wget -y --no-install-recommends

# Install python packages
RUN pip3 install tensorflow-serving-api==1.14.0 grpcio gevent requests opencv-python

# Copy client code
COPY src /src
WORKDIR /src

# Download Raccoon image
RUN mkdir -p /data/images
RUN wget -O /data/images/raccoon.jpg -nv --no-check-certificate "https://raw.githubusercontent.com/experiencor/raccoon_dataset/master/images/raccoon-1.jpg"

# Remove temp and cache folders
RUN rm -rf /var/lib/apt/lists/* && rm -rf /var/cache/apt/* && rm -rf /root/.cache/* && rm -rf /install && apt-get clean