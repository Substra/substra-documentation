FROM python:3.10

WORKDIR /usr/src/app

COPY Makefile .
COPY examples examples/
COPY substrafl_examples substrafl_examples/

RUN pip install -r examples/diabetes_example/assets/requirements.txt \
    -r examples/titanic_example/assets/requirements.txt \
    -r substrafl_examples/get_started/torch_fedavg_assets/requirements.txt \
    -r substrafl_examples/go_further/sklearn_fedavg_assets/requirements.txt

    