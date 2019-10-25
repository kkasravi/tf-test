#!/usr/bin/env bash
if [[ -z $PYENV_ROOT ]]; then
  source $HOME/.profile
fi
pyenv install 3.7.4
pyenv virtualenv 3.7.4 tpu
pyenv local tpu
pyenv global tpu
pip install pipenv
pipenv install absl-py
pipenv install pyyaml
pipenv install wheel
pipenv install tensorflow==1.14
pipenv install google-cloud-storage
pipenv install google-api-python-client
pipenv install oauth2client
