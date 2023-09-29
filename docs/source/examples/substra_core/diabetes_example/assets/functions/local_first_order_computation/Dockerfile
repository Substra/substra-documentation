# this base image works in both CPU and GPU enabled environments
FROM ghcr.io/substra/substra-tools:0.20.0-nvidiacuda11.8.0-base-ubuntu22.04-python3.9

# install dependencies
RUN pip3 install pandas numpy

# add your algorithm script to docker image
ADD federated_analytics_functions.py .

# define how script is run
ENTRYPOINT ["python3", "federated_analytics_functions.py", "--function-name", "local_first_order_computation"]
