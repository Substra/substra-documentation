FROM python:3.11-slim

WORKDIR /usr/src/app

COPY  substra-tools/ substra-tools/
COPY  substra/ substra/
COPY  substrafl/ substrafl/

RUN cd substrafl && python -m pip install --no-cache-dir -e .

RUN cd substra && python -m pip install --no-cache-dir -e .

RUN cd substra-tools && python -m pip install --no-cache-dir -e .

COPY substra-documentation/Makefile substra-documentation/
COPY substra-documentation/examples_requirements.txt substra-documentation/
COPY substra-documentation/docs/source/examples substra-documentation/docs/source/examples/

RUN cd substra-documentation && make install-examples-dependencies

WORKDIR /usr/src/app/substra-documentation
