# Use an official Python runtime as a parent image
# We choose a specific version (e.g., 3.10-slim-buster) for stability
FROM python:3.10-slim-buster

# Set the working directory in the container
WORKDIR /app

# Install system dependencies if needed (e.g., for some ML libraries)
# For now, minimal additions. If you encounter errors later related to
# dependencies like 'blas', 'lapack', etc., you might need to add:
# RUN apt-get update && apt-get install -y --no-install-recommends \
#      build-essential \
#      libblas-dev \
#      liblapack-dev \
#      gfortran \
#      && rm -rf /var/lib/apt/lists/*

# Copy the requirements file into the container
COPY requirements.txt .

# Install any specified Python packages
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of your application code into the container
# This copies the entire 'your_project' content into '/app'
# This now includes app.py and the frontend/ directory
COPY . .

# Create the data, models, and logs directories if they don't exist
# These directories are created by config.py, but explicit creation here
# ensures they are present even if config.py isn't run directly during image build
RUN mkdir -p data models logs

# Define environment variables (optional, but good practice)
ENV PYTHONUNBUFFERED 1

# Expose the port Flask will run on
EXPOSE 5000

# Command to run your application
# This is a placeholder. The 'run.sh' script will typically override this
# with a specific command like 'python train.py'
CMD ["python", "--version"]