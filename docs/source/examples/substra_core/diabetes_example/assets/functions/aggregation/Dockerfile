# this base image works in both CPU and GPU enabled environments
FROM python:3.12-slim

# install dependencies
RUN pip3 install pandas numpy substratools

# add your algorithm script to docker image
ADD federated_analytics_functions.py .

# define how script is run
ENTRYPOINT ["python3", "federated_analytics_functions.py", "--function-name", "aggregation"]
