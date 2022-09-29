# this base image works in both CPU and GPU enabled environments
<<<<<<< HEAD
FROM gcr.io/connect-314908/substra-tools:0.18.0-nvidiacuda11.6.0-base-ubuntu20.04-python3.9
=======
FROM ghcr.io/substra/substra-tools:0.16.0-nvidiacuda11.6.0-base-ubuntu20.04-python3.9
>>>>>>> 2ec0c9a (chore: release 0.21.0 (#189))

# install dependencies
RUN pip3 install numpy==1.23.1 scikit-learn==1.1.1 torch==1.11.0

# add your metrics script to docker image
ADD metrics.py .

# define how script is run
ENTRYPOINT ["python3", "metrics.py", "--method-name", "score"]
