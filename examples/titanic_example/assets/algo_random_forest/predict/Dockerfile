# this base image works in both CPU and GPU enabled environments
FROM ghcr.io/substra/substra-tools:0.18.0-nvidiacuda11.8.0-base-ubuntu22.04-python3.9

# install dependencies
RUN pip3 install pandas numpy 'scikit-learn==0.24.2' pillow scipy keras

# add your algorithm script to docker image
ADD titanic_algo_rf.py .

# define how script is run
ENTRYPOINT ["python3", "titanic_algo_rf.py", "--function-name", "predict"]
