# this base image works in both CPU and GPU enabled environments
FROM python:3.12-slim

# install dependencies
RUN pip3 install pandas numpy 'scikit-learn==1.5.2' substra

# add your function script to docker image
ADD titanic_function_rf.py .

# define how script is run
ENTRYPOINT ["python3", "titanic_function_rf.py", "--function-name", "predict"]
