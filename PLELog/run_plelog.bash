#!/bin/bash

# Run the Docker container with a volume mount
docker run -it -v "$(pwd)/PLELog:/app/PLELog" -v "/home/mohammad/projects/CallGraphPruner/data:/app/data" plelog
