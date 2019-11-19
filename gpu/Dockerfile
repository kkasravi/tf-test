FROM tensorflow/tensorflow:latest-gpu-py3

RUN apt-get update && apt-get install -y \
    vim \
    git && \
    curl https://pyenv.run | bash && \
    echo "set -o vi" >> $HOME/.profile && \
    echo "export LC_ALL='en_US.UTF-8'" >> $HOME/.profile && \
    echo 'export PYENV_ROOT="$HOME/.pyenv"' >> $HOME/.profile && \
    echo 'export PIPENV_PYTHON="$PYENV_ROOT/shims/python"' >> $HOME/.profile && \
    echo 'export PIPENV_VERBOSITY=-1' >> $HOME/.profile && \
    echo 'export PATH="$PYENV_ROOT/bin:$PATH"' >> $HOME/.profile && \
    echo 'eval "$(pyenv init -)"' >> $HOME/.profile && \
    echo 'eval "$(pyenv virtualenv-init -)"' >> $HOME/.profile 

RUN mkdir /kubeflow
WORKDIR /kubeflow
COPY . /kubeflow
RUN pip install packaging
#RUN ./env-model.sh
