FROM gfour/doop:latest

# Install Python 3, pip, and pandas
RUN apt-get update && \
    apt-get install -y python3 python3-pip && \
    pip3 install --no-cache-dir pandas && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# Mount the current directory to /scripts
VOLUME /scripts

# Mount the benchmark to /benchmark
VOLUME /benchmark

# Mount the data directory to /data
VOLUME /data

# Default shell
CMD ["/bin/bash"]
