# this base image works in both CPU and GPU enabled environments
FROM ghcr.io/substra/substra-tools:0.19.0-nvidiacuda11.8.0-base-ubuntu22.04-python3.9

# install dependencies
RUN pip3 install scikit-learn==1.0.2

# add your metrics script to docker image
ADD titanic_metrics.py .

# define how script is run
ENTRYPOINT ["python3", "titanic_metrics.py", "--function-name", "score"]
