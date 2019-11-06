#!/usr/bin/env bash
gcloud container clusters create tpu-models-cluster --cluster-version=1.14.7 --scopes=cloud-platform --enable-ip-alias --enable-tpu --machine-type=n1-standard-8 --min-cpu-platform='Intel Skylake' --zone=us-central1-b
