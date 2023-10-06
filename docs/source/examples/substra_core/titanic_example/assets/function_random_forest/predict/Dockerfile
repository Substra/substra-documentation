# this base image works in both CPU and GPU enabled environments
FROM ghcr.io/substra/substra-tools:0.21.0-nvidiacuda11.8.0-base-ubuntu22.04-python3.9

# install dependencies
RUN pip3 install pandas numpy 'scikit-learn==1.1.1'

# add your function script to docker image
ADD titanic_function_rf.py .

# define how script is run
ENTRYPOINT ["python3", "titanic_function_rf.py", "--function-name", "predict"]
