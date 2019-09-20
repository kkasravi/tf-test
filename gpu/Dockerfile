FROM tensorflow/tensorflow:latest-gpu-py3

RUN apt-get update && apt-get install -y vim
WORKDIR /kubeflow
COPY model.py .
CMD ["python", "model.py"]
