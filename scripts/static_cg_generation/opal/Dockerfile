FROM ubuntu:20.04 as opal-build
SHELL ["/bin/bash", "-c"]

USER root

## timezone \
RUN apt-get update && \
    apt-get install -yq tzdata && \
    ln -fs /usr/share/zoneinfo/America/Chicago /etc/localtime && \
    dpkg-reconfigure -f noninteractive tzdata 

## package update
RUN apt-get update && apt-get upgrade -y && \
    apt-get install -y apt-utils git openjdk-8-jdk maven && \
    apt-get install -y python3 python3-pip && \
    pip3 install --no-cache-dir pandas && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*


WORKDIR /
RUN git clone https://github.com/UTD-FAST-Lab/OPALInterface.git
WORKDIR /OPALInterface
RUN git fetch
RUN git checkout new_algorithms
RUN mvn compile package
WORKDIR /

# Default shell
CMD ["/bin/bash"]
