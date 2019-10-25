#!/usr/bin/env bash
if [[ -z $PYENV_ROOT ]]; then
  source $HOME/.profile
fi
pyenv activate tpu
python resnet/resnet_main.py --data_dir=gs://cloud-tpu-test-datasets/fake_imagenet --model_dir=gs://kdkasrav/resnet
