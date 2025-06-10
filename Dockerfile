# Use an official Python runtime as a parent image
FROM python:3.10-slim-buster

# Set the working directory in the container
WORKDIR /app

# Copy the requirements file into the container
COPY requirements.txt .

# Install any specified Python packages
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of your application code into the container
COPY . .

# Expose the port the app runs on
EXPOSE 8000

# Command to run your application's web server
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]