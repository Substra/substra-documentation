FROM python:3.10

WORKDIR /usr/src/app

COPY Makefile .
COPY examples examples/

RUN make install-examples-dependencies

