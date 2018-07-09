FROM python:3.6-alpine

RUN apk update && apk upgrade && \
    apk add --no-cache bash git openssh

RUN pip install numpy

RUN git clone https://github.com/QTIM-Lab/qtim_tools /home/qtim_tools
WORKDIR /home/qtim_tools
RUN python /home/qtim_tools/setup.py develop

WORKDIR "/"