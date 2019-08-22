FROM tensorflow/serving:1.14.0

# Set non-interactive for linux packages installation
ENV DEBIAN_FRONTEND=noninteractive

# Copy script
COPY monitor_serving.sh /usr/bin/monitor_serving.sh

ENTRYPOINT ["/usr/bin/monitor_serving.sh"]