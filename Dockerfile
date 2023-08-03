FROM nvidia/cuda:11.7.1-base-ubuntu20.04

RUN apt-get update \
    && apt-get install -y python3-pip python3-dev \
    && cd /usr/local/bin \
    && ln -s /usr/bin/python3 python \
    && pip3 install --upgrade pip

WORKDIR /home/

COPY requirements.txt /home/itailab/ep20124/InnerSpeech_EEGNet

RUN pip3 install -r requirements.txt

WORKDIR /home/
