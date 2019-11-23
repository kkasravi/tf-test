#!/usr/bin/env bash
if [[ -z $PYENV_ROOT && -f $HOME/.profile ]]; then
  source $HOME/.profile
fi
venv=$(pyenv virtualenvs|grep '^*'|awk '{print $2}')
if [[ $venv != "gpu" ]]; then
  pyenv install 3.6.8
  pyenv virtualenv 3.6.8 gpu
  pyenv local gpu
fi
pip install pipenv
pipenv install absl-py
pipenv install pyyaml
pipenv install wheel
pipenv install tensorflow==1.14
pipenv install google-cloud-storage
pipenv install google-api-python-client
pipenv install oauth2client
pipenv install packaging
