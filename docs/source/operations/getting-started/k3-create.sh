#!/bin/sh
set -e
k3d cluster delete || echo 'No cluster'
mkdir -p /tmp/org-1
mkdir -p /tmp/org-2
k3d cluster create --api-port 127.0.0.1:6443 -p 80:80@loadbalancer -p 443:443@loadbalancer --k3s-arg "--no-deploy=traefik,metrics-server@server:*" --volume /tmp/org-1:/tmp/org-1 --volume /tmp/org-2:/tmp/org-2

# Patch and install nginx-ingress
curl https://raw.githubusercontent.com/kubernetes/ingress-nginx/master/deploy/static/provider/kind/deploy.yaml > /tmp/deploy.yaml
gsed -i 's/        - --publish-status-address=localhost/        - --publish-status-address=localhost\n        - --enable-ssl-passthrough/g' /tmp/deploy.yaml
gsed -i "/ingress-ready: \"true\"/d" /tmp/deploy.yaml
kubectl apply -f /tmp/deploy.yaml

kubectl create ns orderer
kubectl create ns org-1
kubectl create ns org-2
