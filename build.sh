#!/bin/bash

# Define your Docker image name
IMAGE_NAME="fantasy-football-ai"
TAG="latest"

echo "Building Docker image: ${IMAGE_NAME}:${TAG}"

# Build the Docker image
docker build -t "${IMAGE_NAME}:${TAG}" .

if [ $? -eq 0 ]; then
    echo "Docker image built successfully: ${IMAGE_NAME}:${TAG}"
else
    echo "Failed to build Docker image."
    exit 1
fi