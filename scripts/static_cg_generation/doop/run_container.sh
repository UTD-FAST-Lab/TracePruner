#!/bin/bash

IMAGE_NAME="doop-runner"
DOCKERFILE_PATH="./Dockerfile"

# Customize these paths
BENCHMARK_DIR="/20TB/mohammad/xcorpus-total-recall/jarfiles"
DATA_DIR="/20TB/mohammad/xcorpus-total-recall/static_cgs/doop"

# # Check for image existence
# if [[ "$(docker images -q $IMAGE_NAME 2> /dev/null)" == "" ]]; then
#     echo " Image '$IMAGE_NAME' not found. Building it now..."
#     docker build -t $IMAGE_NAME -f "$DOCKERFILE_PATH" .
# else
#     echo " Image '$IMAGE_NAME' already exists. Skipping build."
# fi

# Build the Docker image
echo " Building Docker image..."
docker build -t $IMAGE_NAME -f "$DOCKERFILE_PATH" .

# Run the container
echo " Running container..."
docker run -it --rm \
    -v "$(pwd)":/scripts \
    -v "$BENCHMARK_DIR":/benchmark \
    -v "$DATA_DIR":/data \
    $IMAGE_NAME