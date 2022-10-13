
FROM ghcr.io/substra/substra-tools:0.18.0-nvidiacuda11.6.0-base-ubuntu20.04-python3.9

COPY . .

# install dependencies
RUN python3.9 -m pip install -U pip

# Install substrafl, substra (and substratools if editable mode)


# PyPi dependencies
RUN python3.9 -m pip install --no-cache-dir numpy==1.23.1 torch==1.11.0

# Install local dependencies


ENTRYPOINT ["python3.9", "algo.py", "--function-name", "train"]
