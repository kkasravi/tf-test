# Docker image of TensorBoard and TPU Profiler.
FROM ubuntu:bionic as build_utils
RUN apt-get update && apt-get install -y --no-install-recommends \
        ca-certificates \
        build-essential \
        git \
        python \
        python-pip \
        python-setuptools && \
    pip install tensorflow==1.14 && \
    pip install google-cloud-storage && \
    pip install google-api-python-client && \
    pip install oauth2client && \
    pip install cloud-tpu-profiler==1.14


# Docker image for running TPU tensorflow examples.
FROM ubuntu:bionic as build

RUN apt-get update && apt-get install -y \
        curl \
        wget \
        sudo \
        gnupg \
        lsb-release \
        ca-certificates \
        build-essential \
        zlib1g \
        zlib1g-dev \
        git \
        vim \
        locales \
        libsqlite3-dev \
        sqlite3 \
        bzip2 \
        libssl-dev \
        openssl \
        libgdbm-dev \
        libgdbm-compat-dev \
        liblzma-dev \
        libreadline-dev \
        libncursesw5-dev \
        libffi-dev \
        uuid-dev && \
    locale-gen "en_US.UTF-8" && \
    update-locale LC_ALL="en_US.UTF-8" && \
    export CLOUD_SDK_REPO="cloud-sdk-$(lsb_release -c -s)" && \
    echo "deb https://packages.cloud.google.com/apt $CLOUD_SDK_REPO main" > /etc/apt/sources.list.d/google-cloud-sdk.list && \
    curl https://packages.cloud.google.com/apt/doc/apt-key.gpg | apt-key add - && \
    apt-get update && \
    apt-get install -y google-cloud-sdk && \
    curl https://pyenv.run | bash && \
    echo "set -o vi" >> $HOME/.profile && \
    echo "export LC_ALL='en_US.UTF-8'" >> $HOME/.profile && \
    echo 'export PYENV_ROOT="$HOME/.pyenv"' >> $HOME/.profile && \
    echo 'export PIPENV_PYTHON="$PYENV_ROOT/shims/python"' >> $HOME/.profile && \
    echo 'export PIPENV_VERBOSITY=-1' >> $HOME/.profile && \
    echo 'export PATH="$PYENV_ROOT/bin:$PATH"' >> $HOME/.profile && \
    echo 'eval "$(pyenv init -)"' >> $HOME/.profile && \
    echo 'eval "$(pyenv virtualenv-init -)"' >> $HOME/.profile 
    
# Checkout tensorflow/models at the appropriate branch
COPY . /tensorflow_tpu_models
WORKDIR /tensorflow_tpu_models
RUN ./env-model.sh

#RUN git clone -b r1.14 --depth 1 https://github.com/tensorflow/models.git /tensorflow_models

# Checkout tensorflow/tpu at the appropriate branch
#RUN git clone -b r1.14 --depth 1 https://github.com/tensorflow/tpu.git /tensorflow_tpu_models
