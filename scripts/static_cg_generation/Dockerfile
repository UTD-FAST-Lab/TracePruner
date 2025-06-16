FROM ubuntu:20.04


# INSTALL PRELIMINARY TOOLS
RUN apt-get update && \
    DEBIAN_FRONTEND=noninteractive \
    apt-get -y install sudo apt-transport-https ca-certificates software-properties-common apt-utils  \
                       git curl less nano vim zip zlib1g-dev wget python3.5 unzip aptitude make gcc g++ \
                       libtool libncurses5-dev libopenmpi-dev openmpi-bin libsqlite3-dev libffi-dev mcpp \
                       flex bison clang build-essential



# Install OpenJDK 8 version 342
ARG JDK_DIR="/home/${USER}/jdk"
RUN sudo apt-get -y autoremove openjdk* && \
    sudo apt-get -y purge openjdk* && \
    mkdir ${JDK_DIR} && \
    cd ${JDK_DIR} && \
    wget https://github.com/adoptium/temurin8-binaries/releases/download/jdk8u342-b07/OpenJDK8U-jdk_x64_linux_hotspot_8u342b07.tar.gz && \
    sudo tar xvf $JDK_DIR/OpenJDK8U-jdk_x64_linux_hotspot_8u342b07.tar.gz -C /opt && \
    sudo update-alternatives --install /usr/bin/java java /opt/jdk8u342-b07/bin/java 1000 && \
    sudo update-alternatives --install /usr/bin/javac javac /opt/jdk8u342-b07/bin/javac 1000 && \
    sudo update-alternatives --install /usr/bin/jar jar /opt/jdk8u342-b07/bin/jar 1000

## install sbt
RUN echo "deb https://repo.scala-sbt.org/scalasbt/debian all main" | sudo tee /etc/apt/sources.list.d/sbt.list && \
    echo "deb https://repo.scala-sbt.org/scalasbt/debian /" | sudo tee /etc/apt/sources.list.d/sbt_old.list && \
    sudo apt-key adv --keyserver hkps://keyserver.ubuntu.com:443 --recv 2EE0EA64E40A89B84B2DF73499E82A75642AC823 && \
    sudo apt-get update && \
    sudo apt-get install sbt


## install OPAL
ARG OPAL_BRANCH="develop"
ARG OPAL_COMMIT_ID="fa37a36181cdcb3135999e76ffe90e617a434f8e"
ARG OPAL_DIR="/home/$USER/OPAL"
RUN git clone https://github.com/stg-tud/opal.git ${OPAL_DIR} && \
    cd ${OPAL_DIR} && \
    git checkout ${OPAL_BRANCH} && \
    git reset --hard ${OPAL_COMMIT_ID} && \
    sbt publishLocal


## install JCG
ARG JCG_BRANCH="feature/DynCG"
ARG JCG_COMMIT_ID="a5dc5649fbfff38142db77996854d9e195edf14a"
ARG JCG_DIR="/home/$USER/JCG"
RUN git clone https://github.com/opalj/JCG.git $JCG_DIR && \
    cd $JCG_DIR && \
    git checkout feature/DynCG && \
    git reset --hard ${JCG_COMMIT_ID} && \
    sbt compile

WORKDIR $JCG_DIR


# Default shell
CMD ["/bin/bash"]
