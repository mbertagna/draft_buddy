# Base image for all Docker Compose services.
FROM python:3.10-slim-buster

# Repository root inside the container.
WORKDIR /app

# Copy package metadata and source first for dependency installation.
COPY pyproject.toml .
COPY README.md .
COPY src/ ./src/

# Install project dependencies and the package itself.
RUN pip install --no-cache-dir .

# Copy the remaining repository files used by compose-run scripts.
COPY . .

# Ensure runtime output directories exist on fresh images.
RUN mkdir -p data models logs

# Default runtime environment for script-based entrypoints.
ENV PYTHONUNBUFFERED 1
ENV PYTHONPATH=/app/src

# FastAPI webapp default port.
EXPOSE 5001

# Compose services override the default command with the appropriate script.
CMD ["python", "--version"]
