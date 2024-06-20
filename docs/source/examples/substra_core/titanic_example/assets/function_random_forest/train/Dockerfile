# this base image works in both CPU and GPU enabled environments
FROM python:3.11-slim

# install dependencies
RUN pip3 install pandas 'numpy<2.0' 'scikit-learn==1.5.0' substratools

# add your function script to docker image
ADD titanic_function_rf.py .

# define how script is run
ENTRYPOINT ["python3", "titanic_function_rf.py", "--function-name", "train"]
