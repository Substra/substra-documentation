Orchestrator deployment
=======================

We start from a fully working kubernetes cluster

Entrypoint for helm deployment: https://artifacthub.io/packages/helm/substra/orchestrator

Add the helm repository for Substra:
```
helm repo add substra https://substra.github.io/charts/
```

Create a values.yaml file and set the hostname of your orchestrator

```
ingress:
  enabled: true
  hostname: orc.my-corp.org
```

We want to validate clients identity usigng mTLS so we need to enable some additional settings:
In the same values file add:

```
orchestrator:
  verifyClientMSPID: true
```

We enable TLS for the server:
in te same values.yaml file
```
orchestrator:
  verifyClientMSPID true
  tls:
    enabled: true
```

Now for this to work we need to generate some certificates.

First we need to generate a CA certificate:
for this example we will generate them manually but you can use external providers like let's encrypt for this step.

> Include sample openssl config: https://github.com/Substra/orchestrator/blob/9c8106dde71ae379da0a8ae4d0bb0e8d88b68f4d/examples/tools/openssl-with-ca.cnf

```
openssl genrsa -out ca.key 2048
openssl req -new -x509 -days 365 -sha256 -key ca.key -extensions v3_ca -config openssl-with-ca.cnf -subj "/C=FR/ST=Loire-Atlantique/L=Nantes/O=Orchestrator Root CA/CN=Orchestrator Root CA" -out ca.crt
```

You should now have these files in your current directory: `ca.crt` and `ca.key`.

You can already create a ConfigMap in you cluster named `orchestrator-tls-cacert` using the command:
```
kubectl create configmap orchestrator-tls-cacert --from-file=ca.crt
```

> list configmap to see that it is created

Generate a certificate key for the orchestrator
```
openssl req -newkey rsa:2048 -nodes -keyout orchestrator-tls.key -subj "/C=FR/ST=Loire-Atlantique/L=Nantes/O=Substra/CN=orc.my-corp.org" -out orchestrator-cert.csr
```

and 
```
openssl x509 -req -days 365 -in orchestrator-cert.csr -CA ca.crt -CAkey ca.key -CAcreateserial -out orchestrator-tls.crt -extfile <(printf "subjectAltName=DNS:orc.my-corp.org")
```

Then we need to create a kubernetes secret containing these informations

```
kubectl create secret tls orchestrator-tls-server-pair --cert=orchestrator-tls.crt --key=orchestrator-tls.key
```

Now you can run an orchestrator that is secured

```
helm install my-orchestrator substra/orchestrator --version 7.4.3 --values orchestrator-values.yaml
```

