FROM ultralytics/ultralytics:latest

ENV DEBIAN_FRONTEND=noninteractive

RUN --mount=type=cache,sharing=locked,target=/var/cache/apt --mount=type=cache,sharing=locked,target=/var/lib/apt \
    apt-get update && apt-get install --no-install-recommends -qqy \
    bash-completion \
    x11-common \
    x11-utils

RUN pip3 install opencv-python

# # Change HOME environment variable
ENV HOME=/home

COPY .config/entrypoint.sh /ros_entrypoint.sh
ENTRYPOINT ["/ros_entrypoint.sh"]
CMD ["bash"]