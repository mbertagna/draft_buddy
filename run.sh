#!/bin/bash

# Define your Docker image name
IMAGE_NAME="fantasy-football-ai"
TAG="latest"

echo "Starting interactive session in Docker container for ${IMAGE_NAME}:${TAG}"
echo "Your current directory is mounted at /app"

# Run the Docker container, mounting the current directory and dropping you into a shell
# This provides an interactive environment with access to your local files.
docker run \
    -it \
    --rm \
    -p 8000:8000 \
    -v "$(pwd):/app" \
    "${IMAGE_NAME}:${TAG}" \
    /bin/bash