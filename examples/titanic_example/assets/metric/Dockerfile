# this base image works in both CPU and GPU enabled environments
FROM gcr.io/connect-314908/substra-tools:0.14.0-nvidiacuda11.6.0-base-ubuntu20.04-python3.9

# install dependencies
RUN pip3 install scikit-learn==1.0.2

# add your metrics script to docker image
ADD titanic_metrics.py .

# define how script is run
ENTRYPOINT ["python3", "titanic_metrics.py", "--method-name", "score"]
