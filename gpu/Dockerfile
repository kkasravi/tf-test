FROM tensorflow/tensorflow:latest-gpu-py3

RUN apt-get update && apt-get install -y vim
RUN mkdir /tf
WORKDIR /tf
COPY model.py .
CMD ["python", "model.py"]
