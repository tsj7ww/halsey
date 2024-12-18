#!/bin/bash

# Make script executable and fail on error
set -e

# Clone the repository if it doesn't exist
if [ ! -d "predml" ]; then
    echo "Cloning repository..."
    git clone https://github.com/yourusername/predml.git
    cd predml
else
    cd predml
    echo "Repository already exists, pulling latest changes..."
    git pull
fi

# Build development container
echo "Building development container..."
docker-compose build predml-dev

# Start container in interactive mode
echo "Starting development environment..."
docker-compose run --rm predml-dev