# TODO these tagging is not working
# Use build args for image metadata
ARG IMAGE_NAME=language_discovery_pruning
ARG IMAGE_VERSION=0.1

# Use the official Python 3.10 image as the base image
FROM python:3.10-slim

LABEL org.opencontainers.image.title="${IMAGE_NAME}" \
      org.opencontainers.image.version="${IMAGE_VERSION}" \
      org.opencontainers.image.vendor="n30tri8" \
      org.opencontainers.image.description="Language discovery pruning project" \
      org.opencontainers.image.source="https://github.com/n30tri8/language_discovery_pruning"

# Set the working directory in the container
WORKDIR /app

# Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the project files into the container
COPY . .

# Command to run the application
ENTRYPOINT ["python", "main.py"]
