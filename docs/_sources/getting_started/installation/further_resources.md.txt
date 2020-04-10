# Further resources

## Need help?

Let's talk:

- [WIP] [Create an issue on Github](https://github.com/SubstraFoundation/substra/issues/new)
- Come chat with us on [Slack](https://substra-workspace.slack.com/archives/CT54J1U2E) (Once your request is granted, you will be able to join us, especially the *#help* channel)
- Have a look to the [forum](https://forum.substra.org/)
- Drop us an [email](mailto:help@substra.ai)
- Or come meet us *irl* in Paris, Nantes or Limoges!

## Further resources

### K8s

- `kubectx` & `kubens`: <https://github.com/ahmetb/kubectx#installation>
- Local Kubernetes deployment with minikube: <https://kubernetes.io/blog/2019/03/28/running-kubernetes-locally-on-linux-with-minikube-now-with-kubernetes-1.14-support/>
- [Awesome Kubernetes list](https://github.com/ramitsurana/awesome-kubernetes#starting-point)
- [Minikube](https://minikube.sigs.k8s.io/) is recommended on Ubuntu but you can also use [Microk8s](https://microk8s.io/).

### K9s

Here are some [k9s](https://github.com/derailed/k9s) tips:

- `CTRL + A`
- `:xray deployments all`
- `?` for help
- `/server` then `l` for the logs
- `:jobs` might be useful to see what is happening behind the scene
- `y` to see the YAML configuration

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

## Acknowledgements

This amazing piece of software has been developed and open sourced by [Owkin](https://owkin.com/) and its [terrific software engineers](https://github.com/SubstraFoundation/substra/graphs/contributors). The repositories are now maintained by [Substra Foundation](https://github.com/SubstraFoundation) and its community. Besides, Substra is really excited to welcome new members, feedbacks and contributions, so please, feel free to get in touch with us!