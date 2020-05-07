# Further resources

## Troubleshooting

### Virtualization issues

- If you are getting this error about libvirt: `[KVM_CONNECTION_ERROR] machine in unknown state: getting connection: getting domain: error connecting to libvirt socket`. You probably need to install additional package: `sudo apt-get install libvirt-daemon-system`
- You might need to change the owner as well: `sudo chown -R $USER:$USER $HOME/.kube` `sudo chown -R $USER:$USER $HOME/.minikube`; See <https://medium.com/@nieldw/running-minikube-with-vm-driver-none-47de91eab84c>

### Kubectl useful commands

- `kubectl cluster-info`
- `kubectl get all --all-namespaces`
- `kubectl delete ns YOUR_NAMESPACE`

### Minikube useful commands

- `minikube ip`
- `minikube dashboard`
- `minikube tunnel`
- `minikube config view`
- `minikube addons list`
- If you are using microk8s:
  - `microk8s.status`
  - `microk8s.inspect`

### Tiller

- Tiller might need you to use this command in case of error during init: `helm init --service-account tiller --upgrade`. You can also try to create a service account with `kubectl create serviceaccount --namespace kube-system tiller`. Otherwise, please have a look here: <https://github.com/SubstraFoundation/substra-backend/pull/1>
- tiller issues: <https://stackoverflow.com/questions/51646957/helm-could-not-find-tiller#51662259>
- After running `skaffold dev` in the `hlf-k8s` repo, in case of error related to the `tempchart` folder, please do `rm -rf tempchart`

### Virtualization resources

- [KVM (Kernel Virtual Machine) installation](https://help.ubuntu.com/community/KVM/Installation#Installation)
- Required packages: [Ubuntu help](https://help.ubuntu.com/community/KVM/Installation#Install_Necessary_Packages)
- If you need more information about [libvirt & qemu](https://libvirt.org/drvqemu.html)

### Http errors

- If you are getting a `403` error only on <http://substra-backend.node-1.com/> and <http://substra-frontend.node-1.com/> with Firefox, please check if `dns over https` is activated (in Firefox Network options). If so, please try again desactivating this option, or try with another browser...
- If you are getting `bad certificate` issues: please try to investigate your setup with `helm list` or `helm list --all`; you can try `helm delete network-org-1-peer-1 --no-hooks` & in k9s `:jobs` and delete the `orgs` & `orderer`; you can also`helm delete --purge RELEASE_NAME` (ex. `network-org-1-peer-1`) and then restart with `skaffold dev`.
- `Self-signed certificate` issues are related to your network provider/admin

### Serve the frontend with Yarn

Alternatively, instead of using `skaffold`, you might want to start the `substra-frontend` with [Yarn](https://yarnpkg.com/getting-started/install):

Start Redis in one terminal window:

```sh
redis-cli
```

Launch Yarn in another terminal window:

```sh
yarn install

API_URL=http://substra-backend.node-2.com yarn start
```

You will then have to map the frontend urls to your localhost, like this:

```sh
127.0.0.1 substra-frontend.node-1.com substra-frontend.node-2.com
```

You can now head to <http://substra-frontend.node-2.com:3000/> and start to play with Substra!

### Backend & Browser extension

In order to use the backend webpage on your browser, you will need to install this extension that will send a special header containing a `version`:

- [Firefox](https://addons.mozilla.org/en-US/firefox/addon/modheader-firefox/)
- [Chrome](https://chrome.google.com/webstore/detail/modheader/idgpnmonknjnojddfkpgkljpfnnfcklj)

You will then need add these three elements:

| Resource | Name | Value |
| -------- | ---- | ----- |
| Request Headers | `Accept` | `text/html;version=0.0, */*; version=0.0` |
| Filters | `URL Pattern` | `http://susbtra-backend.node-1.com` |
| Filters | `URL Pattern` | `http://susbtra-backend.node-2.com` |

Otherwise, you can try to import the following configuration to the extension (via the Import menu):

```json
[
    {
        "title": "Profile 1",
        "hideComment": true,
        "headers": [
            {
                "enabled": true,
                "name": "Accept",
                "value": "text/html;version=0.0, */*; version=0.0",
                "comment": ""
            }
        ],
        "respHeaders": [],
        "filters": [
            {
                "enabled": true,
                "type": "urls",
                "urlRegex": "http://substra-backend.node-2.com"
            },
            {
                "enabled": true,
                "type": "urls",
                "urlRegex": "http://susbtra-backend.node-1.com"
            }
        ],
        "urlReplacements": [],
        "appendMode": false
    }
]
```

See: <https://github.com/SubstraFoundation/substra-backend#testing-with-the-browsable-api>

## Tips and useful resources

### K8s

- `kubectx` & `kubens`: <https://github.com/ahmetb/kubectx#installation>
- Local Kubernetes deployment with minikube: <https://kubernetes.io/blog/2019/03/28/running-kubernetes-locally-on-linux-with-minikube-now-with-kubernetes-1.14-support/>
- [Awesome Kubernetes list](https://github.com/ramitsurana/awesome-kubernetes#starting-point)
- [Minikube](https://minikube.sigs.k8s.io/) is recommended on Ubuntu but you can also use [Microk8s](https://microk8s.io/).
- Use k8s context:
  - `kubectl config get-contexts`
  - `kubectl config set current-context YOUR_CONTEXT`

### K9s

[k9s](https://github.com/derailed/k9s) is quite handy to inspect your Kubernetes setup. Here are some basic tips:

- `?` for help
- `y` to see the `YAML` configuration
- `d` for `describe`
- `l` for `logs`
- `/server` then `l` for the logs
- `:jobs` might be useful to see what is happening behind the scene
- `CTRL` + `A`
- `:xray deployments all`

### Helm

- Use `helm ls` to get the list of your helm releases (packages). You can also use commands like `helm delete NAME_OF_THE_CHART`
- <https://www.linux.com/tutorials/helm-kubernetes-package-manager/>
- [Substra Helm charts](https://hub.helm.sh/charts/substra/hlf-k8s)
- [Helm 2 documentation](https://v2.helm.sh/docs/)
- [Helm general file structure](https://v2.helm.sh/docs/developing_charts/#the-chart-file-structure)

### Hyperledger Fabric

- Installation: <https://medium.com/hackernoon/hyperledger-fabric-installation-guide-74065855eca9#c566>
- Building your first network: <https://hyperledger-fabric.readthedocs.io/en/release-1.4/build_network.html>

### Python Virtual Environment

In order to keep your installation of Substra separated from your general Python environement, which is a general Python good practice, it is recommanded to prepare a Python [virtual environment](https://virtualenv.pypa.io/en/latest/). In a new terminal window, please use one of the following method:

```sh
# Method 1: install the virtualenv package
pip3 install --user virtualenv

# Create a new virtual environment
virtualenv -p python3 NAME_OF_YOUR_VENV
# or even
virtualenv -p $(which python3) NAME_OF_YOUR_VENV

# Method 2: install the python3-venv package
sudo apt install python3-venv # (Ubuntu)

# Create a new virtual environment
python3 -m venv NAME_OF_YOUR_VENV

# Method 1 & 2: activate your new virtual env
source NAME_OF_YOUR_VENV/bin/activate

# Method 1 & 2: install Substra package inside your fresh new virtual environment
pip3 install substra

# Method 1 & 2: stop your virtual environment
deactivate
```

If you are looking for more Python Virtual Environment resource, you might be interested in this post from [Real Python](https://realpython.com/python-virtual-environments-a-primer/).

## Get in touch

- Come chat with us on [Slack](https://substra-workspace.slack.com/archives/CT54J1U2E) (Once your request is granted, you will be able to join us, especially the *#help* channel)
- Drop us an [email](mailto:help@substra.ai)
- [Create an issue on Github](https://github.com/SubstraFoundation/substra/issues/new)
- Or come meet us *irl* in Paris, Nantes or Limoges!

## Acknowledgements

This amazing piece of software has been developed and open sourced by [Owkin](https://owkin.com/) and its [terrific software engineers](https://github.com/SubstraFoundation/substra/graphs/contributors). The repositories are now maintained by [Substra Foundation](https://github.com/SubstraFoundation) and its community. Besides, Substra is really excited to welcome new members, feedbacks and contributions, so please, feel free to get in touch with us!