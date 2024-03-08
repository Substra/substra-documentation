# this base image works in both CPU and GPU enabled environments
FROM python:3.9-slim

# install dependencies
RUN pip3 install pandas numpy 'scikit-learn==1.1.1'

# add your metrics script to docker image
ADD titanic_metrics.py .

# define how script is run
ENTRYPOINT ["python3", "titanic_metrics.py", "--function-name", "score"]
