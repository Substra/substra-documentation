# Local installation of Substra using Kubernetes and Skaffold

> This is an ongoing document, please feel free to reach us or to [raise any issue](https://github.com/SubstraFoundation/substra-documentation/issues).

This guide will help you run the Substra platform on your machine in development mode, with a two nodes setup.

- [Compatibility table](#compatibility-table)
- [Substra Setup](#substra-setup)
  - [General knowledge](#general-knowledge)
  - [Hardware requirements](#hardware-requirements)
  - [Software requirements](#software-requirements)
    - [Kubernetes](#kubernetes)
      - [Running Kubernetes locally](#running-kubernetes-locally)
      - [Installing Kubectl](#installing-kubectl)
    - [Helm](#helm)
    - [Skaffold](#skaffold)
  - [Virtualization](#virtualization)
  - [Get the source code (Mac & Ubuntu)](#get-the-source-code-mac--ubuntu)
  - [Configuration](#configuration)
    - [Minikube (Ubuntu)](#minikube-ubuntu)
    - [Helm init (Mac & Ubuntu)](#helm-init-mac-ubuntu)
    - [Network](#network)
- [Running the platform](#running-the-platform)
  - [Start Substra](#start-substra)
    - [1. hlf-k8s repository](#1-hlf-k8s-repository)
    - [2. substra-backend repository](#2-substra-backend-repository)
    - [3. substra-frontend repository](#3-substra-frontend-repository)
  - [Stop Substra](#stop-substra)
  - [Reset Substra](#reset-substra)
- [Login, password and urls](#login-password-and-urls)
  - [Credentials and urls](#credentials-and-urls)
  - [Substra CLI config & login (Mac & Ubuntu)](#substra-cli-config--login-mac--ubuntu)

___

When everything is ready, you will be able to start Substra with:

```sh
# If you use Minikube
minikube start --cpus 6 --memory 8192 --disk-size 50g --kubernetes-version='v1.16.7'

# In 3 different terminal windows, in this order:
# In the repository hlf-k8s
SUBSTRA_HLF_VERSION=0.0.16
git checkout $SUBSTRA_HLF_VERSION
skaffold deploy --images substrafoundation/fabric-tools:$SUBSTRA_HLF_VERSION --images substrafoundation/fabric-peer:$SUBSTRA_HLF_VERSION

# In the repository substra-backend
SUBSTRA_BACKEND_VERSION=0.1.6
git checkout $SUBSTRA_BACKEND_VERSION
skaffold deploy --images substrafoundation/substra-backend:$SUBSTRA_BACKEND_VERSION

# In the repository substra-frontend
SUBSTRA_FRONTEND_VERSION=0.0.20
git checkout $SUBSTRA_FRONTEND_VERSION
skaffold deploy --images substrafoundation/substra-frontend:$SUBSTRA_FRONTEND_VERSION
```

## Compatibility table

Please always refer to the [compatibility table](https://github.com/SubstraFoundation/substra#compatibility-table) and use the suggested releases for each section.

## Substra Setup

### General knowledge

In order to install Substra, it is *recommended* to be comfortable with your package manager and to have some basic knowledge of Unix (Mac or Ubuntu/Debian) administration and network. It might also be useful to have a good knowledge of Docker containers and Kubernetes orchestration.

### Hardware requirements

> Please be cautious with the parameters passed to Kubernetes as it might try to use all the allocated resources without regards for your system!

If you wish to comfortably run Substra, it is advised to have:

- **50 GB of free space** as several images need to be pulled.
- **8 GB of RAM** for Kubernetes.

### Software requirements

> Note: Please **always** refer to the package provider website before installing any software!
> Note: Please use LTS versions of Ubuntu to ensure best compatibility (18.04 & 20.04).

Substra deployment is orchestrated by `Kubernetes` and `Minikube` is a great tool for your local Kubernetes deployments. For Mac users, we recommend to use Docker Desktop with Kubernetes, but Minikube is an alternative.

`Helm` is a Kubernetes package manager that helps install and manage Kubernetes applications. `Tiller` runs inside the Kubernetes Cluster and manages releases of your charts. Helm has two parts : helm (client) & tiller (server). Charts are Helm packages that contains a package description (`Chart.yaml`) and one or more templates which contain Kubernetes manifest files. Charts can be stored locally or on a distant repository.

`Skaffold` is a program working on top of the Kubernetes configuration that will operate the deployment for you. It relies on the `skaffold.yaml` files that you will find at the root of each repositories.

#### Kubernetes

##### Running Kubernetes locally

- Mac

First of all, download the `Docker desktop` installer from <https://www.docker.com/products/docker-desktop>. You'll have to create an account there to do so. Then run it to install Docker on your machine. Once installed, launch Docker and open its "preferences" panel. In the Kubernetes tab, check the `Enable Kubernetes` checkbox. If you want, you can select Minikube from the Docker toolbar and restart Docker. Kubernetes will take a while to launch the first time, but once it is done, you can move on to configuring.

- Ubuntu: [Docker](https://docs.docker.com/engine/install/ubuntu/) & [Minikube](https://minikube.sigs.k8s.io/docs/start/)

```sh
# Ubuntu only
# Get the executable & install it
curl -LO https://storage.googleapis.com/minikube/releases/latest/minikube-linux-amd64
sudo install minikube-linux-amd64 /usr/local/bin/minikube
```

##### Installing [Kubectl](https://kubernetes.io/docs/tasks/tools/install-kubectl/)

- Mac

```sh
brew install kubectl
```

- Ubuntu

Please use **[Kubectl v1.16.7](https://github.com/kubernetes/kubectl/releases/tag/v0.16.7)**:

```sh
sudo apt-get update && sudo apt-get install -y apt-transport-https
curl -s https://packages.cloud.google.com/apt/doc/apt-key.gpg | sudo apt-key add -
echo "deb https://apt.kubernetes.io/ kubernetes-xenial main" | sudo tee -a /etc/apt/sources.list.d/kubernetes.list
sudo apt-get update
sudo apt-get install -y kubectl=1.16.7-00 -V
```

#### [Helm](https://helm.sh/)

To install Helm v3, please refer to the [official documentation](https://v3.helm.sh/docs/intro/install/).

Note: substra up to version 0.6.0 only supports helm v2.

#### [Skaffold](https://skaffold.dev/)

- Mac

```sh
brew install skaffold
```

- Ubuntu

```sh
# Get the executable (1.8 version)
curl -Lo skaffold https://storage.googleapis.com/skaffold/releases/v1.8.0/skaffold-linux-amd64
# Make it executable on your machine
chmod +x skaffold
# Move it to your local bin
sudo mv skaffold /usr/local/bin
```

### Virtualization

- If you use Mac, this virtualization part is already handled for you.
- If you are in a virtualized environment (*within a Virtual Machine*), you will have to:
  - install `socat` with the command `sudo apt-get install socat`
  - launch all commands with `sudo`
  - pass the parameter `--vm-driver=none` when starting Minikube (`minikube start (...)`)
- If you use Ubuntu (*not in a VM*), you will need to:
  - Validate your host virtualization with the command `virt-host-validate`, see [this for further resources](https://linux.die.net/man/1/virt-host-validate)

### Get the source code (Mac & Ubuntu)

> Note 1: As Hyperledger Fabric is a permissioned blockchain, ordering nodes are in charge of the transaction ordering, see [Fabric documentation](https://hyperledger-fabric.readthedocs.io/en/release-1.4/orderer/ordering_service.html)
>
> Note 2: Please refer to the [compatibility table](https://github.com/SubstraFoundation/substra#compatibility-table) and use the relevant releases. For example, in the `substra-backend` repository, use `git checkout 0.0.14`. You can also only clone a single specific branch/release with the `--single-branch` option, for example: `git clone https://github.com/SubstraFoundation/substra.git --single-branch --branch 0.5.0`.

You will find the main Substra repository [here](https://github.com/SubstraFoundation/substra), but in order to run the Substra framework, you will need to clone 3 repositories: [hlf-k8s](https://github.com/SubstraFoundation/hlf-k8s) (Hyperledger Fabric), [substra-backend](https://github.com/SubstraFoundation/substra-backend) and [substra-frontend](https://github.com/SubstraFoundation/substra-frontend).

The `hlf-k8s` repository is in charge of the initialization of the [Hyperledger Fabric](https://www.hyperledger.org/projects/fabric) network. By default, it will create an `orderer` and two orgs (`org-1` & `org-2`).

The `substra-backend` is powered by Django and is responsible for, among other things, handling the api endpoints.

The `substra-frontend` will serve a neat interface for the end-users.

Go to the folder where you want to add the repositories and launch this command:

```sh
git clone https://github.com/SubstraFoundation/hlf-k8s.git
git clone https://github.com/SubstraFoundation/substra-frontend.git
git clone https://github.com/SubstraFoundation/substra-backend.git

# Optional
git clone https://github.com/SubstraFoundation/substra.git
```

> Note: if you do not have `git` on your machine, you can also download and unzip in the same folder the code using these links:
>
> - [hlf-k8s](https://github.com/SubstraFoundation/hlf-k8s/archive/master.zip)
> - [substra-backend](https://github.com/SubstraFoundation/substra-backend/archive/master.zip)
> - [substra-frontend](https://github.com/SubstraFoundation/substra-frontend/archive/master.zip)

### Configuration

#### Minikube (Ubuntu)

> Note: If you are using Mac, this part will be handled by Docker Desktop for you; you can directly head to the Helm section. Still, you can use Minikube on Mac and select it in your Docker Desktop application.

Please enable the ingress minikube module: `minikube addons enable ingress`. You might need to edit `skaffold.yaml` files and set `nginx-ingress.enabled` to `false`.

You can now start Minikube with:

```sh
# Comfortable setup
minikube start --cpus 6 --memory 8192 --disk-size 50g --kubernetes-version='v1.16.7'
# Frugal setup
minikube start --cpus 4 --memory 8192 --disk-size 30g --kubernetes-version='v1.16.7'
# VM setup (Inside a VM, you will have to execute all commands with sudo)
sudo minikube start --vm-driver=none --kubernetes-version='v1.16.7'
```

#### Helm init (Mac & Ubuntu)

Substra uses charts coming from different helm repositories, which you need to add:

```sh
helm repo add bitnami https://charts.bitnami.com/bitnami
helm repo add owkin https://owkin.github.io/charts/
helm repo add stable https://charts.helm.sh/stable
```

#### Network

We will now configure your machine so that the urls `substra-backend.node-1.com`, `substra-backend.node-2.com`, `substra-frontend.node-1.com` and `substra-frontend.node-2.com` point to your local instance of the platform.

- Ubuntu

```sh
# Append your Kubernetes cluster ip to your system hosts
echo "$(minikube ip) substra-backend.node-1.com substra-frontend.node-1.com substra-backend.node-2.com substra-frontend.node-2.com" | sudo tee -a /etc/hosts

# Inside a VM, you will need to use
echo "$(sudo minikube ip) substra-backend.node-1.com substra-frontend.node-1.com substra-backend.node-2.com substra-frontend.node-2.com" | sudo tee -a /etc/hosts
```

Example:

```sh
192.168.39.32 substra-backend.node-1.com substra-frontend.node-1.com substra-backend.node-2.com substra-frontend.node-2.com
```

If you want to customize your configuration,
you can assign the ingress loadbalancer ip to the corresponding url, for example:

```sh
10.111.206.49 substra-backend.node-1.com substra-frontend.node-1.com
10.109.77.31 substra-backend.node-2.com substra-frontend.node-2.com
```

If you want to expose another service, you can use something like:

```sh
kubectl port-forward -n ${NAMESPACE} service/${SERVICE_NAME} ${SERVICE_PORT}:${PORT_WANTED}
```

- Mac

You'll first need to get the host IP address from container by running:

```sh
docker run -it --rm busybox ping host.docker.internal
```

The output should look like this:

```sh
Unable to find image 'busybox:latest' locally
latest: Pulling from library/busybox
0f8c40e1270f: Pull complete
Digest: sha256:1303dbf110c57f3edf68d9f5a16c082ec06c4cf7604831669faf2c712260b5a0
Status: Downloaded newer image for busybox:latest
PING host.docker.internal (192.168.65.2): 56 data bytes
64 bytes from 192.168.65.2: seq=0 ttl=37 time=0.426 ms
64 bytes from 192.168.65.2: seq=1 ttl=37 time=1.356 ms
64 bytes from 192.168.65.2: seq=2 ttl=37 time=1.187 ms
64 bytes from 192.168.65.2: seq=3 ttl=37 time=1.870 ms
64 bytes from 192.168.65.2: seq=4 ttl=37 time=3.638 ms
```

With new `64 bytes from 192.168.65.2: seq=4 ttl=37 time=3.638 ms` lines added every second. Hit `Ctrl-c` to stop it.

Please note that you may not see `192.168.65.2` but another address. In this case, you'll have to update the following commands with your address.

Then run:

```sh
sudo ifconfig lo0 alias 192.168.65.2
```

You'll be asked for your password. Once done, you can check that the command ran successfully by running:

```sh
ifconfig lo0
```

The output should look like this:

```sh
lo0: flags=8049<UP,LOOPBACK,RUNNING,MULTICAST> mtu 16384
options=1203<RXCSUM,TXCSUM,TXSTATUS,SW_TIMESTAMP>
inet 127.0.0.1 netmask 0xff000000
inet6 ::1 prefixlen 128
inet6 fe80::1%lo0 prefixlen 64 scopeid 0x1
inet 192.168.65.2 netmask 0xffffff00
nd6 options=201<PERFORMNUD,DAD>
```

The last step will be to update `/etc/hosts` by adding this new line to it:

```sh
192.168.65.2 substra-backend.node-1.com substra-frontend.node-1.com substra-backend.node-2.com substra-frontend.node-2.com
```

Running the following command will do it for you (you'll be asked for your password again):

```sh
echo "192.168.65.2 substra-backend.node-1.com substra-frontend.node-1.com substra-backend.node-2.com substra-frontend.node-2.com" | sudo tee -a /etc/hosts
```

## Running the platform

> Please refer to the [compatibility table](https://github.com/SubstraFoundation/substra#compatibility-table) and use the relevant releases. For example, in the `substra-backend` repository, use `git checkout 0.0.14`. You can also only clone a single specific branch/release with the `--single-branch` option, for example: `git clone https://github.com/SubstraFoundation/substra.git --single-branch --branch 0.5.0`.

> note: These instructions assume you want to run a specific substra version for evaluation purpose.
> This is **not** the development process, for which you should refer to the readme documents of each project.

### Start Substra

#### Use Helm

You can use `helm` to directly install Substra components. You will find an handy [usage section](https://helm.sh/docs/intro/using_helm/) directly on their website.

Before starting, please refer to the [helm hub](https://artifacthub.io/packages/search?page=1&repo=substra) to get the charts version you want to install:

1. [hlf-k8s](https://artifacthub.io/packages/helm/substra/hlf-k8s)
2. [substra-backend](https://artifacthub.io/packages/helm/substra/substra-backend)
3. [substra-frontend](https://artifacthub.io/packages/helm/substra/substra-frontend)

Then, we will have to inspect and adjust to your needs (or just use it as it is for demo purpose) the different files located in the `values` folder and referenced in the root `skaffold.yaml` file with the `valuesFiles` parameter.

You will find several configurations in the `/example` folder of the `hlf-k8s` repository, organized as:

```sh
# 2-orgs-policy-any example
├── skaffold.yaml
└── values
    ├── orderer.yaml
    ├── org-1-peer-1.yaml
    └── org-2-peer-1.yaml

# 4-orgs-policy-majority example
├── skaffold.yaml
└── values
    ├── orderer.yaml
    ├── org-1-peer-1.yaml
    ├── org-2-peer-1.yaml
    ├── org-3-peer-1.yaml
    └── org-4-peer-1.yaml
```

When you are ready to go, you will have to install the different charts with helm. 
For example, following the [compatibility table](https://github.com/SubstraFoundation/substra#compatibility-table), to install a [substra cli v0.8.0](https://github.com/SubstraFoundation/substra/releases/tag/0.8.0) compatible server:

```sh
# hlf-k8s orderer
helm install orderer --namespace orderer substra/hlf-k8s --version 5.1.0 # -f values/orderer.yaml

# hlf-k8s org-1
helm install org-1-peer-1 --namespace org-1 substra/hlf-k8s --version 5.1.0 # -f values/org-1-peer-1.yaml

# hlf-k8s org-2
helm install org-2-peer-1 --namespace org-2 substra/hlf-k8s --version 5.1.0 # -f values/org-2-peer-1.yaml

# substra-backend org-1
helm install backend-org-1 --namespace org-1 substra/substra-backend --version 1.6.0 # -f values/backend-org-1.yaml

# substra-backend org-2
helm install backend-org-2 --namespace org-2 substra/substra-backend --version 1.6.0 # -f values/backend-org-2.yaml

# substra-frontend org-1
helm install frontend-org-1 --namespace org-1 substra/substra-frontend --version 1.0.0-alpha.2 # -f values/frontend-org-1.yaml

# substra-frontend org-2
helm install frontend-org-2 --namespace org-2 substra/substra-frontend --version 1.0.0-alpha.2 # -f values/frontend-org-2.yaml
```

#### Or just use skaffold

> Note: Please be aware that these commands are quite long to be executed and might take a few minutes, especially for the first installation.

On both Mac and Ubuntu, once your Kubernetes cluster is up and running (started via Minikube or Docker) and Tiller initialized, you will need to start Substra with Skaffold.

#### 1. hlf-k8s repository

In the `hlf-k8s` repository, run the following commands:

```sh
SUBSTRA_HLF_VERSION=0.0.16
git checkout $SUBSTRA_HLF_VERSION
skaffold deploy --images substrafoundation/fabric-tools:$SUBSTRA_HLF_VERSION --images substrafoundation/fabric-peer:$SUBSTRA_HLF_VERSION --tail
```

This will checkout the appropriate code revision and install the deployment in your local cluster.
The docker images will be pulled from the official registry.

The platform will be ready once the terminal displays:

```sh
[network-org-2-peer-1-hlf-k8s-chaincode-install-0-4bdd4 fabric-tools] 2019-11-14 09:14:52.070 UTC [chaincodeCmd] install -> INFO 003 Installed remotely response:<status:200 payload:"OK" >
# or
[network-org-2-peer-1-hlf-k8s-channel-join-0-kljgq fabric-tools] 2020-02-10 10:18:02.211 UTC [channelCmd] InitCmdFactory -> INFO 001 Endorser and orderer connections initialized
# or
[network-org-2-peer-1-hlf-k8s-channel-join-0-kljgq fabric-tools] 2020-02-10 10:18:02.350 UTC [channelCmd] executeJoin -> INFO 002 Successfully submitted proposal to join channel
```

![status:200 payload:"OK"](../img/start_hlf-k8s.png "status:200 payload:'OK'")

You can stop the log output by pressing `ctrl-c`, this won't stop hlf-k8s.

#### 2. substra-backend repository

In the `substra-backend` repository, run the following commands:

```sh
SUBSTRA_BACKEND_VERSION=0.1.6
git checkout $SUBSTRA_BACKEND_VERSION
skaffold deploy --images substrafoundation/substra-backend:$SUBSTRA_BACKEND_VERSION --tail
```

The platform will be ready once the terminal displays:

```sh
[backend-org-2-substra-backend-server-74bb8486fb-nkq6m substra-backend] INFO - 2020-02-10 10:24:42,514 - django.server - "GET /liveness HTTP/1.1" 200 2
# or
[backend-org-1-substra-backend-server-77cf8cb9fd-cwgs6 substra-backend] INFO - 2020-02-10 10:24:51,393 - django.server - "GET /readiness HTTP/1.1" 200 2
```

![django.server readiness](../img/start_backend.png "django.server readiness")

You can stop the log output by pressing `ctrl-c`, this won't stop substra-backend.

#### 3. substra-frontend repository

In the `substra-frontend` repository, run the following commands:

```sh
SUBSTRA_FRONTEND_VERSION=0.0.20
git checkout $SUBSTRA_FRONTEND_VERSION
skaffold deploy --images substrafoundation/substra-frontend:$SUBSTRA_FRONTEND_VERSION --tail
```

The platform will be ready once the terminal displays:

```sh
[frontend-org-2-substra-frontend-787554fc4b-pmh2g substra-frontend] CACHING:  /login
```

![CACHING Login](../img/start_frontend.png "CACHING Login")

You can stop the log output by pressing `ctrl-c`, this won't stop substra-backend.

Alternatively, instead of using `skaffold`, you might want to start the `substra-frontend` with [Yarn](https://yarnpkg.com/getting-started/install).
If you want to do, please refer to [this section](#serve-the-frontend-with-yarn).

### Stop Substra

In order to stop Substra, issue the command `skaffold delete` in each repository.
On Ubuntu, if you want to stop the minikube Kubernetes cluster, you can use `minikube stop`.

If you want to restart, you will just need to run again the `skaffold deploy` command in the 3 repositories.

### Reset Substra

You can reset your installation (if you've used `skaffold deploy`) with:

```sh
# run from each repositories (hlf-k8s, substra-backend, substra-frontend)
skaffold delete
# or
kubectl rm ns org-1 org-2 orderer
# On Ubuntu, to remove all the Kubernetes cluster
minikube delete
```

## Login, password and urls

### Credentials and urls

Once the platform is running, you can sign in to the two organizations using the default development credentials:

org-1:

- API url: `http://substra-backend.node-1.com`
- Frontend url: `http://substra-frontend.node-1.com`
- Username: `node-1`
- Password: `p@$swr0d44`

org-2:

- API url: `http://substra-backend.node-2.com`
- Frontend url: `http://substra-frontend.node-2.com`
- Username: `node-2`
- Password: `p@$swr0d45`

You should find the credentials in the charts: `skaffold.yaml` files or in the `substra-backend/charts/substra-backend/values.yaml`.

### Substra CLI config & login (Mac & Ubuntu)

> Note 1: Substra works on Python 3.6.
>
> Note 2: If you are working inside a virtualized environment, you probably will have to execute `pip3` commands with `sudo`.

Install the CLI:

> Need help to setup a Python Virtual Environment? [Check this out](further_resources.md#python-virtual-environment)

```sh
pip3 install substra
```

Login with the CLI

```sh
# Configuration
substra config --profile node-1 http://substra-backend.node-1.com

# Login
substra login --profile node-1 --username node-1 --password 'p@$swr0d44'

# Then you can try
substra list node --profile node-1

# And you can always get help
substra  --help
```

This is it, if the `substra login` command exited successfully, you're good to go!

Congratulations \o/

If you want to go further, please refer to:

- Documentation:
  - [CLI](https://github.com/SubstraFoundation/substra/blob/master/references/cli.md#summary)
  - [SDK](https://github.com/SubstraFoundation/substra/blob/master/references/sdk.md#substrasdk)
- Examples:
  - [Titanic](https://github.com/SubstraFoundation/substra/blob/master/examples/titanic/README.md#titanic)
  - [Cross-validation](https://github.com/SubstraFoundation/substra/blob/master/examples/cross_val/README.md#cross-validation)
  - [Compute plan](https://github.com/SubstraFoundation/substra/blob/master/examples/compute_plan/README.md#compute-plan)
