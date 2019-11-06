#!/usr/bin/env bash
if [[ -z $PYENV_ROOT ]]; then
  source $HOME/.profile
fi
pyenv activate tpu
pipenv run python mnist/mnist_main.py --data_dir=gs://cloud-tpu-test-datasets/fake_imagenet --model_dir=gs://kdkasrav/resnet

  --tpu=$TPU_NAME \
  --data_dir=${STORAGE_BUCKET}/data \
  --model_dir=${STORAGE_BUCKET}/mnist \
  --use_tpu=True \
  --iterations=500 \
  --train_steps=2000
