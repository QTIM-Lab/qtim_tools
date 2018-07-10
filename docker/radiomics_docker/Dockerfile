FROM python:2.7

RUN pip install --no-cache-dir numpy scipy matplotlib

RUN echo 3

RUN git clone https://github.com/QTIM-Lab/qtim_tools /home/qtim_tools
WORKDIR /home/qtim_tools

RUN git pull
RUN python /home/qtim_tools/setup.py develop

WORKDIR "/"