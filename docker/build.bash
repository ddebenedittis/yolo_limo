#!/bin/bash

# Navigate to the root directory of the project.
cd "$(dirname "$(dirname "$(realpath "${BASH_SOURCE[0]}")")")"

mkdir -p detect

docker build \
    -t yolo_limo \
    -f docker/Dockerfile .