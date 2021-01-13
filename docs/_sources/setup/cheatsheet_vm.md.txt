# Cheatsheet Ubuntu 20.04 VM

> /!\ This setup is only compatible with Substra v.0.6.0. It will soon be updated.
> Please refer to the [compatibility table](https://github.com/SubstraFoundation/substra#compatibility-table) for further information.

> /!\ This configuration is totally not suitable for production environment!

## System

### Hardware check

Please check first the hardware-related information of your machine and update accordingly installation procedures. This guide aims at `x86_64` machines.

```sh
uname -mp
```

### System updates

```sh
sudo apt update && \
    sudo apt upgrade -y && \
	sudo apt autoremove --purge -y && \
	sudo apt autoclean
```

## Softwares installation

### [Docker](https://docs.docker.com/engine/install/ubuntu/)

```sh
sudo apt-get install \
    apt-transport-https \
    ca-certificates \
    curl \
    gnupg-agent \
    software-properties-common \
    socat # needed by k8s/skaffold in virtualized context

curl -fsSL https://download.docker.com/linux/ubuntu/gpg | sudo apt-key add -

sudo add-apt-repository \
    "deb [arch=amd64] https://download.docker.com/linux/ubuntu \
    $(lsb_release -cs) \
    stable"

sudo apt-get update && \
    sudo apt-get install docker-ce docker-ce-cli containerd.io

sudo groupadd docker
sudo usermod -aG docker $USER
```

### [Minikube v1.9.2](https://minikube.sigs.k8s.io/docs/start/)

```sh
curl -LO https://github.com/kubernetes/minikube/releases/download/v1.9.2/minikube-linux-amd64 && \
    sudo install minikube-linux-amd64 /usr/local/bin/minikube && \
    rm -rf minikube-linux-amd64
```

### [Kubectl v1.16.7](https://github.com/kubernetes/kubectl/releases/tag/v0.16.7)

```sh
curl -s https://packages.cloud.google.com/apt/doc/apt-key.gpg | sudo apt-key add - && \
    echo "deb https://apt.kubernetes.io/ kubernetes-xenial main" | sudo tee -a /etc/apt/sources.list.d/kubernetes.list

sudo apt-get update && \
    sudo apt-get install -y kubectl=1.16.7-00 -V
```

### [Helm v2.16.1](https://github.com/helm/helm/releases/tag/v2.16.1)

```sh
curl -LO https://get.helm.sh/helm-v2.16.1-linux-amd64.tar.gz && \
    tar xzvf helm-v2.16.1-linux-amd64.tar.gz && \
    sudo mv linux-amd64/tiller linux-amd64/helm /usr/local/bin/ && \
    rm -rf helm-v2.16.1-linux-amd64.tar.gz linux-amd64
```

### [Skaffold v1.8.0](https://skaffold.dev/)

```sh
curl -Lo skaffold https://storage.googleapis.com/skaffold/releases/v1.8.0/skaffold-linux-amd64 && \
    chmod +x skaffold && \
    sudo mv skaffold /usr/local/bin
```

### [k9s v0.22.0](https://github.com/derailed/k9)

> Note: this one is not mandatory but it is such a great k8s tool that I couldn't resist to add it!

```sh
mkdir k9s && cd k9s && \
    curl -L0 https://github.com/derailed/k9s/releases/download/v0.22.0/k9s_Linux_x86_64.tar.gz && \
    tar xzvf k9s_Linux_x86_64.tar.gz && \
    sudo mv k9s /usr/local/bin && \
    rm -rf k9s
```

## Configuration

### Minikube config

```sh
sudo minikube start --vm-driver=none --kubernetes-version='v1.16.7'
```

In case of `HOST_JUJU_LOCK_PERMISSION` error, please update as follow:

```sh
sudo sysctl fs.protected_regular=0
```

Enable addons

```sh
# Check minikube status
sudo minikube status

# Check the enabled addons
sudo minikube addons list

# You'll need at least ingress & helm-tiller addons activated
sudo minikube addons enable ingress && \
    sudo minikube addons enable ingress-dns && \
    sudo minikube addons enable helm-tiller
```

### Helm init

```sh
sudo helm init && \
    sudo kubectl get pods --namespace kube-system | grep tiller
```

### Add bitnami repo to Helm repos

```sh
sudo helm repo add bitnami https://charts.bitnami.com/bitnami
```

## Network quick config

### On the server

```sh
echo "$(sudo minikube ip) substra-backend.node-1.com substra-frontend.node-1.com substra-backend.node-2.com substra-frontend.node-2.com" | sudo tee -a /etc/hosts
```

### On your local machine (Mac & Linux)

Set the server external ip in the `/etc/hosts` file.

> Please replace `<SERVER-EXTERNAL-IP>` with the external ip of your server.

```sh
echo "<SERVER-EXTERNAL-IP> substra-backend.node-1.com substra-frontend.node-1.com substra-backend.node-2.com substra-frontend.node-2.com" | sudo tee -a /etc/hosts
```

## Clone the repositories and switch to the correct version

Please refer to the [compatibility table](https://github.com/SubstraFoundation/substra#compatibility-table).

> Note: this is a setup compatible with substra v0.6.0

```sh
mkdir substra && cd substra
git clone https://github.com/SubstraFoundation/substra.git --branch 0.6.0
git clone https://github.com/SubstraFoundation/hlf-k8s.git --branch 0.0.12 # or checkout fe33fc0 ¯\_(ツ)_/¯
git clone https://github.com/SubstraFoundation/substra-frontend.git --branch 0.0.17
git clone https://github.com/SubstraFoundation/substra-backend.git --branch 0.0.19
```

## Fire up

> Note: In hlf-k8s repository, you will need to update the `skaffold.yaml` file and edit the `chaincodes[0].src` variable for each organisation.
> You will have to use the correct chaincode version according to the [compatibility table](https://github.com/SubstraFoundation/substra#compatibility-table), for example, with substra v0.6.0:
> `chaincodes[0].src: https://github.com/SubstraFoundation/substra-chaincode/archive/0.0.10.tar.gz`

```sh
# start hlf-k8s
cd hlf-k8s && sudo skaffold run

# start the backend
cd ../substra-backend && sudo skaffold run

# start the frontend
cd ../substra-frontend && sudo skaffold run

# Quick test on backend node-1
# From the server & from your local machine:
curl substra-backend.node-1.com/readiness
# should return "OK"
```

If you didn't see `OK`:

- Check that your `/etc/hosts` file is using the correct ip and that domains are registered
- Use [k9s](https://github.com/derailed/k9s)
- Check the [troubleshooting section](https://doc.substra.ai/setup/further_resources.html)
- We also have a [#help](https://substra-workspace.slack.com/archives/CT54J1U2E) channel on Slack
