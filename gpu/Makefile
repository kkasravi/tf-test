IMG ?= gcr.io/constant-cubist-173123/tf-test-gpu
TAG ?= $(eval TAG := $(shell git describe --tags --long --always))$(TAG)

all: build

auth:
	docker login  -u "user" -p "password" docker.io

build:
	DOCKER_BUILDKIT=1 docker build --tag $(IMG):$(TAG) .
	@echo Built $(IMG):$(TAG)

push: build
	#docker push kkasravi/tf-test-gpu:3b8d4aa
	docker push $(IMG):$(TAG)
	@echo Pushed $(IMG):$(TAG)
