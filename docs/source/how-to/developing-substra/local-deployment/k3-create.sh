#!/bin/sh
set -e

if [[ "$OSTYPE" == "darwin"* ]]; then
    SED_EXEC="gsed"
else
    SED_EXEC="sed"
fi

k3d cluster delete || echo 'No cluster'
mkdir -p /tmp/org-1
mkdir -p /tmp/org-2
mkdir -p /tmp/org-3
k3d cluster create --api-port 127.0.0.1:6443 -p 80:80@loadbalancer -p 443:443@loadbalancer --k3s-arg "--disable=traefik,metrics-server@server:*" --volume /tmp/org-1:/tmp/org-1 --volume /tmp/org-2:/tmp/org-2 --volume /tmp/org-3:/tmp/org-3

# Patch and install nginx-ingress
curl https://raw.githubusercontent.com/kubernetes/ingress-nginx/master/deploy/static/provider/kind/deploy.yaml > /tmp/deploy.yaml
$SED_EXEC -i 's/        - --publish-status-address=localhost/        - --publish-status-address=localhost\n        - --enable-ssl-passthrough/g' /tmp/deploy.yaml
$SED_EXEC -i "/ingress-ready: \"true\"/d" /tmp/deploy.yaml
kubectl apply -f /tmp/deploy.yaml
kubectl create ns orderer

# Create namespaces and apply PSA baseline label
for org_index in 1 2 3
do
    org_name="org-${org_index}"
    kubectl create ns ${org_name}
    kubectl label ns ${org_name} pod-security.kubernetes.io/enforce=baseline
done
