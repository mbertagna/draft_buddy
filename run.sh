#!/bin/bash

# Define your Docker image name
IMAGE_NAME="fantasy-football-ai"
TAG="latest"

# Define the command to run inside the container
# We now run our Flask web application
COMMAND="python app.py"
# NOW we execute our main training script!
# COMMAND="python train.py"
# COMMAND="python simulate.py"


echo "Running Docker container for ${IMAGE_NAME}:${TAG}"

# Check if the data, models, and logs directories exist locally
# If not, create them to ensure consistent volume mounts
mkdir -p data models logs

# Run the Docker container, publishing port 5000 from container to host
docker run \
    -it \
    --rm \
    -p 8000:8000 \
    -v "$(pwd)/data:/app/data" \
    -v "$(pwd)/models:/app/models" \
    -v "$(pwd)/logs:/app/logs" \
    "${IMAGE_NAME}:${TAG}" \
    /bin/bash -c "${COMMAND}"

if [ $? -eq 0 ]; then
    echo "Container finished successfully."
else
    echo "Container exited with an error."
    exit 1
fi