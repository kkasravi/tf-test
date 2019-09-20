IMG=gcr.io/constant-cubist-173123/tf-test-clx
TAG ?= $(eval TAG := $(shell git describe --tags --long --always))$(TAG)

all: build

build:
	DOCKER_BUILDKIT=1 docker build --tag $(IMG):$(TAG) .
	@echo Built $(IMG):$(TAG)

push: build
	docker push $(IMG):$(TAG)
	@echo Pushed $(IMG):$(TAG)
