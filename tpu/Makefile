# Copyright 2017 The Kubernetes Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
GCLOUD_PROJECT ?= constant-cubist-173123
TPU ?= gcr.io/$(GCLOUD_PROJECT)/tpu
TPU_UTIL ?= gcr.io/$(GCLOUD_PROJECT)/tpu-util
TAG ?= $(eval TAG := $(shell git describe --tags --long --always))$(TAG)

all: build build_utils

venv:
	./env-model.sh

build:
	DOCKER_BUILDKIT=1 docker build --target build --tag $(TPU):$(TAG) .
	@echo Built $(TPU):$(TAG)

build_utils:
	DOCKER_BUILDKIT=1 docker build --target build_utils --tag $(TPU_UTIL):$(TAG) .
	@echo Built $(TPU_UTIL):$(TAG)

# Build but don't attach the latest tag. This allows manual testing/inspection of the image
# first.
push: build build_utils
	docker push $(TPU):$(TAG)
	@echo Pushed $(TPU):$(TAG)
	docker push $(TPU_UTIL):$(TAG)
	@echo Pushed $(TPU_UTIL):$(TAG)

push-latest: push
	gcloud container images add-tag --quiet $(TPU):$(TAG) $(TPU):latest --verbosity=info
	echo created $(TPU):latest
	gcloud container images add-tag --quiet $(TPU_UTIL):$(TAG) $(TPU_UTIL):latest --verbosity=info
	echo created $(TPU_UTIL):latest

run: build
	capture_tpu_profile --tpu=$(TPU_NAME) --logdir=$(MODEL_BUCKET) --duration_ms=$(TRACING_DURATION_IN_MS)


