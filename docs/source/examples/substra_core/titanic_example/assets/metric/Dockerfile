# this base image works in both CPU and GPU enabled environments
FROM python:3.12-slim

# install dependencies
RUN pip3 install pandas 'numpy<2.0' 'scikit-learn==1.5.0' 'substratools==0.22.0a2'

# add your metrics script to docker image
ADD titanic_metrics.py .

# define how script is run
ENTRYPOINT ["python3", "titanic_metrics.py", "--function-name", "score"]
