#!/usr/bin/env bash
pyenv activate tpu
python resnet/resnet_main.py --data_dir=gs://cloud-tpu-test-datasets/fake_imagenet --model_dir=gs://kdkasrav/resnet
