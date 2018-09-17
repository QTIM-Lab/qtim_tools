FROM python:2.7

RUN pip install --no-cache-dir numpy scipy matplotlib

RUN git clone https://github.com/QTIM-Lab/qtim_tools /home/qtim_tools
WORKDIR /home/qtim_tools

RUN echo 6

RUN git pull
RUN python /home/qtim_tools/setup.py develop

# Commands at startup.
WORKDIR "/"
RUN chmod 777 /home/qtim_tools/docker/entrypoint.sh
ENTRYPOINT ["/home/qtim_tools/docker/entrypoint.sh"]