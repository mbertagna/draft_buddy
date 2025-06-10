#!/bin/bash
IMAGE_NAME="fantasy-football-ai"
TAG="latest"

echo "Running DRAFT BUDDY application..."

# Create directories if they don't exist
mkdir -p data models logs

# Run the Docker container, mapping port 8000
# and mounting volumes to persist data
docker run \
    -it \
    --rm \
    -p 8000:8000 \
    -v "$(pwd)/data:/app/data" \
    -v "$(pwd)/models:/app/models" \
    -v "$(pwd)/logs:/app/logs" \
    "${IMAGE_NAME}:${TAG}"