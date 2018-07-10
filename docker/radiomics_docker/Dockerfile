FROM python:2.7

# RUN apk update && apk upgrade && \
    # apk add --no-cache bash git openssh

RUN apt-get update && apt-get install -y git openssh

RUN pip install --no-cache-dir numpy scipy matplotlib

RUN echo 2

RUN git clone https://github.com/QTIM-Lab/qtim_tools /home/qtim_tools
WORKDIR /home/qtim_tools
RUN python /home/qtim_tools/setup.py develop

WORKDIR "/"