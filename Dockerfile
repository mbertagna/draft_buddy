# Use an official Python runtime as a parent image
# We choose a specific version (e.g., 3.10-slim-buster) for stability
FROM python:3.10-slim-buster

# Set the working directory in the container
WORKDIR /app

# Copy the project configuration files
COPY pyproject.toml .
COPY README.md .
COPY src/ ./src/

# Install dependencies and the project
RUN pip install --no-cache-dir .

# Copy the rest of your application code (api, scripts, frontend, etc.)
COPY . .

# Ensure data directories exist
RUN mkdir -p data models logs

# Define environment variables
ENV PYTHONUNBUFFERED 1
ENV PYTHONPATH=/app/src

# Expose the port Flask will run on
EXPOSE 5001

# Command to run your application
# This is a placeholder. The 'run.sh' script will typically override this
# with a specific command like 'python train.py'
CMD ["python", "--version"]