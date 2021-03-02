# Further resources

## Hands on Substra

If you are facing issues with Substra (CLI or SDK), you can have a look at:

- [Github issues](https://github.com/SubstraFoundation/substra/issues)
- [CLI documentation](https://github.com/SubstraFoundation/substra/blob/master/references/cli.md#summary)
- [SDK documentation](https://github.com/SubstraFoundation/substra/blob/master/references/sdk.md#substrasdk)
- [Debugging](https://doc.substra.ai/debugging.html)
- [Official examples](https://github.com/SubstraFoundation/substra/tree/master/examples#substra-examples) (shipped with substra package)
- [Community examples](https://github.com/SubstraFoundation/substra-examples#substra-examples) (using datasets with images or even video files)

### If you are having issues with:

- A model that cannot be found by the platform: it can be related with the way your model is saved. For example, depending on the assets you are using, the `save_weights()` method from keras might require a path and a file extension:

```python
# replace
model.save_weights(path)
# with
model.save_weights(path, save_format="h5")
# make sure the path and filename are correct
os.rename(path+'.h5', path)
```

> If you didn't find an appropriate answer, you can join our Slack [#help](https://substra.us18.list-manage.com/track/click?e=2effed55c9&id=fa49875322&u=385fa3f9736ea94a1fcca969f) channel!

## Troubleshooting

### Reset

> Caution: Please be aware that theses commands will delete some resources. Be sure to know what you are doing before using it!

If you need to reset and want to start a new kubernetes configuration, you can try some of these:

- `minikube delete`
- `skaffold delete`
- `helm reset` or `helm reset --force`
- `helm delete --purge network-org-1-peer-1 --no-hooks`
- `kubectl delete all -l app=helm -n kube-system`
- `kubectl delete deployment tiller-deploy --namespace kube-system`
- `kubectl delete ns org-1 org-2`

### Kubectl useful commands

- `kubectl cluster-info`
- `kubectl get all --all-namespaces`
- `kubectl delete ns <YOUR_NAMESPACE>`
- `kubectl get nodes -o wide`
- `kubectl get pods -o wide`

### Minikube useful commands

> Check that `helm-tiller` & `ingress` minikube modules are enabled with `minikube addons list`.

- `minikube ip`
- `minikube dashboard`
- `minikube tunnel`
- `minikube config view`
- If you are using microk8s:
  - `microk8s.status`
  - `microk8s.inspect`
- Start minikube with `--alsologtostderr -v=8` to get logs

### Minikube Errors

If you are having this error: `[JUJU_LOCK_DENIED] Failed to start none bare metal machine. "minikube start" may fix it. boot lock: unable to open /tmp/juju-[...] permission denied`, you will need to execute this command: `sudo sysctl fs.protected_regular=0` before restarting minikube.

### Tiller

- Tiller might need you to use this command in case of error during init: `helm init --service-account tiller --upgrade`. You can also try to create a service account with `kubectl create serviceaccount --namespace kube-system tiller`. Otherwise, please have a look here: <https://github.com/SubstraFoundation/substra-backend/pull/1>
- tiller issues: <https://stackoverflow.com/questions/51646957/helm-could-not-find-tiller#51662259>
- After running `skaffold dev` in the `hlf-k8s` repo, in case of error related to the `tempchart` folder, please do `rm -rf tempchart`

### Virtualization issues & resources

- If you are getting this error about `libvirt`: `[KVM_CONNECTION_ERROR] machine in unknown state: getting connection: getting domain: error connecting to libvirt socket`. You probably need to install additional package: `sudo apt-get install libvirt-daemon-system`
- You might need to change the owner as well: `sudo chown -R $USER:$USER $HOME/.kube` `sudo chown -R $USER:$USER $HOME/.minikube`; See <https://medium.com/@nieldw/running-minikube-with-vm-driver-none-47de91eab84c>
- [KVM (Kernel Virtual Machine) installation](https://help.ubuntu.com/community/KVM/Installation#Installation)
- Required packages: [Ubuntu help](https://help.ubuntu.com/community/KVM/Installation#Install_Necessary_Packages)
- If you need more information about [libvirt & qemu](https://libvirt.org/drvqemu.html)

### Http errors

- If you are getting a `403` error only on <http://substra-backend.node-1.com/> and <http://substra-frontend.node-1.com/> with Firefox, please check if `dns over https` is activated (in Firefox Network options). If so, please try again deactivating this option, or try with another browser...
- If you are getting `bad certificate` issues: please try to investigate your setup with `helm list` or `helm list --all`; you can try `helm delete network-org-1-peer-1 --no-hooks` & in k9s `:jobs` and delete the `orgs` & `orderer`; you can also`helm delete --purge RELEASE_NAME` (ex. `network-org-1-peer-1`) and then restart with `skaffold dev`.
- `Self-signed certificate` issues are related to your network provider/admin

## Unreachable backend

If you are having trouble accessing the backends, at some point, you might want to try:

```sh
minikube addons disable ingress
[wait 30 sec]
minikube addons enable ingress
[wait a couple minutes]
curl $(minikube ip) -H 'Host: substra-backend.node-1.com'
```

### Serve the frontend with Yarn

Instead of using `skaffold`, you might want to start the Substra frontend with [Yarn](https://yarnpkg.com/getting-started/install):

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
  - `kubectl config set current-context <YOUR_CONTEXT>`

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

In order to keep your installation of Substra separated from your general Python environment, which is a general Python good practice, it is recommended to prepare a Python [virtual environment](https://virtualenv.pypa.io/en/latest/). In a new terminal window, please use one of the following method:

```sh
# Method 1: install the virtualenv package
pip3 install --user virtualenv

# Create a new virtual environment
virtualenv -p python3 <NAME_OF_YOUR_VENV>
# or even
virtualenv -p $(which python3) <NAME_OF_YOUR_VENV>

# Method 2: install the python3-venv package
sudo apt install python3-venv # (Ubuntu)

# Create a new virtual environment
python3 -m venv <NAME_OF_YOUR_VENV>

# Method 1 & 2: activate your new virtual env
source <NAME_OF_YOUR_VENV>/bin/activate

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
