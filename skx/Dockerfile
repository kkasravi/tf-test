FROM tensorflow/tensorflow:latest-py3

RUN apt-get update && apt-get install -y vim
RUN mkdir /kubeflow
WORKDIR /kubeflow
COPY model.py .
CMD ["python", "model.py"]
