# Demo

Check this [page](https://www.substra.ai/en/demo) and get in touch with us, we will then provide you with instructions and credentials!

## Pre-requisites & Overview of the demo

### Terminology

To follow this tutorial for using the demo instance, it is assumed you have read the documentation of the Substra Framework and are familiar with its core concepts (See [Glossary](https://doc.substra.ai/glossary.html)). In particular the following terms: organisation, asset, algo, objective, metric, opener, train task, test task.

### Technical pre-requisites

The demo instance is deployed on a cloud infrastructure. To use it you will need:

- the credentials that was provided to you after you contacted us for accessing the demo instance
- access to a computer (e.g. a laptop, a VM...) with an internet connection, on which you have `sudo` privileges. Preferably not a Windows machine, although this is still manageable (see below).

### Overview: what you can do with the demo instance

Although this can be inferred from the below tutorial by experienced readers, it might be good to be explicit about the proposed demo:

- Substra Framework is meant to be instantiated as a distributed application among multiple partners. However, for this simple demo to be accessible without any deployment operation and allow you to focus on the data scientist perspective, it is installed as a single instance on a cloud VM, in which 2 `organisations` have been created and live independently.
- The credentials provided to you grant you access to only one of these 2 `organisations`. You will not be able to access or tamper with any asset of the second one; you can consider it as another company to which you don't have access.
- From the organisation you have access to, you will be able to define and register ML tasks to be executed by the distributed application on the assets you targeted. This assumes the task you register are compatible with the permissions that the other organisation defined on its assets (which is what is configured, to make the demo simple).
- A standard example, that we suggest, is the following:
  1. from your organisation you define and register a learning algorithm adapted to the dataset owned by the other organisation. In a real use case, this assumes you were able to discuss with the other organisation about this data science project, and align on the expected dataset pre-processing
  1. you register a task to execute your algorithm on the dataset of the other organisation to train your model
  1. you register a task to test the performance of your model on a test dataset (which can be owned by the other organisation or by yours, that depends on the use case you consider)
  1. you consult the performance recorded in the application distributed ledger
  1. you fetch the model for your own use

## 1. Configuration

### Install the Substra client

> Substra is not yet compatible with Windows. However you should be able to make it work with the Windows Linux Subsystem (see [Substra documentation](https://doc.substra.ai/usage.html?highlight=wsl#installation)), or inside a Linux virtual machine.

It will install the [CLI](https://github.com/SubstraFoundation/substra/blob/master/references/cli.md#summary) (interactive command line interface) & the [Python SDK](https://github.com/SubstraFoundation/substra/blob/master/references/sdk.md#substrasdk) (ready to be used in your Python scripts).

```sh
pip install substra
# Or if a specific version is required by the team (0.6.0 in this example)
pip install substra==0.6.0

# Check that Substra is working
substra --version  # Should return (for example): "substra, version 0.6.0"
```

You will also need to install the package [Substra Tools](https://github.com/SubstraFoundation/substra-tools/tree/master/substratools) which contains all assets templates you will need to submit your algo to the Substra network:

```sh
pip install substratools
# Or if a specific version is required by the team (for example, v0.5.0)
pip install substratools==0.5.0
```

You will then be able to use it in your Python code:

```python
import substratools as tools

# Class Opener
class MyOpener(tools.Opener):

# Class Metrics
class MyMetrics(tools.Metrics):

# Class Algo
class MyAlgo(tools.algo.Algo):
```

This section is documented [here](https://github.com/SubstraFoundation/substra-tools/blob/master/docs/api.md).

### Edit your hosts file

As the demo instance is a demo instance, you will have to create a link between the server ip address (that we provided you together with your credentials) and the 2 nodes domains used in the demo:

```sh
# Replace <CHANGE_ME> with the ip that were provided together with your credentials
echo "<CHANGE_ME> substra-backend.node-1.com substra-frontend.node-1.com substra-backend.node-2.com substra-frontend.node-2.com" | sudo tee -a /etc/hosts
```

This will add a line like the following to the file `/etc/hosts` of your machine:

```sh
# 001.002.003.004     substra-backend.node-1.com substra-backend.node-2.com substra-frontend.node-1.com substra-frontend.node-2.com
```

In order to check that your local substra client can reach the demo server, you can use the following command:

```sh
# Quick test on the backend of the node-1
curl substra-backend.node-1.com/readiness
```

You should receive an `"OK"` answer, if not please check again your `/etc/hosts` file. You can also use your browser to visit [substra-frontend.node-1.com](http:substra-frontend.node-1.com) & [substra-frontend.node-2.com](http:substra-frontend.node-2.com) (without https, because this is demo).

### Login

You can now configure your local Substra client, *with the credentials that were provided to access the demo instance*, to connect to the `node-1` which hosts the organisation `org-1`:

```sh
# Configure the client, with a profile, a user, a password and an url
substra config --profile <PROFILE> http://substra-backend.node-1.com

# Login with the profile you created (located in your home folder: ~/.substra)
substra login --profile <PROFILE> --username <USERNAME> --password '<PASSWORD>'

# You can then list the network nodes that are on the Substra network
substra list node --profile <PROFILE>

# And list, for example, the registered datasets in the node you are logged in
substra list dataset --profile <PROFILE>
```

## 2. Follow Titanic example

With this example you will find everything you need to run a first end-to-end example with the tabular data of the Titanic: see this [repository](https://github.com/SubstraFoundation/substra/blob/master/examples/titanic) for further details.

If you feel more comfortable working with local files, you can clone the [Substra repository](https://github.com/SubstraFoundation/substra) to get the full examples located in the `/example` folder.

## 3. Work with your own assets

> **Note**: This is a demo instance, please do not use any sensitive asset on this platform!

The [debugging section](https://github.com/SubstraFoundation/substra/tree/master/examples/debugging) can be of help preparing your asset, please have a look!

### Preparing your dataset

The first step is partitioning your data, and create a test dataset and a train dataset. Following the Titanic example, you will first need to generate data samples (in the `/test_data_samples` and `/train_data_samples` folders respectively) by updating the `generate_data_samples.py` script with the path to you dataset and running it:

```sh
pip install -r scripts/requirements.txt
python3 scripts/generate_data_samples.py
```

### Testing your assets

To test your assets, you will need to prepare the `opener.py`, `metrics.py` and `algo.py` files which both rely on classes imported from the [substratools](https://github.com/SubstraFoundation/substra-tools/tree/master/substratools).

#### Training tasks

```sh
# Train your model with the train_data_samples
python3 assets/algo/algo.py train \
  --debug \
  --opener-path assets/dataset/opener.py \
  --data-samples-path assets/train_data_samples \
  --output-model-path assets/model/model \
  --log-path assets/logs/train.log

# Predict on train_data_samples with your previously trained model
python3 assets/algo/algo.py predict \
  --debug \
  --opener-path assets/dataset/opener.py \
  --data-samples-path assets/train_data_samples \
  --output-predictions-path assets/pred-train.csv \
  --models-path assets/model/ \
  --log-path assets/logs/train_predict.log \
  model

# Calculate the score of your model on train_data_samples predictions
python3 assets/objective/metrics.py \
  --debug \
  --opener-path assets/dataset/opener.py \
  --data-samples-path assets/train_data_samples \
  --input-predictions-path assets/pred-train.csv \
  --output-perf-path assets/perf-train.json \
  --log-path assets/logs/train_metrics.log
```

#### Testing tasks

```sh
# Predict on test_data_samples with your previously trained model
python3 assets/algo/algo.py predict \
  --debug \
  --opener-path assets/dataset/opener.py \
  --data-samples-path assets/test_data_samples \
  --output-predictions-path assets/pred-test.csv \
  --models-path assets/model/ \
  --log-path assets/logs/test_predict.log \
  model

# Calculate the score of your model on test_data_samples predictions
python3 assets/objective/metrics.py \
  --debug \
  --opener-path assets/dataset/opener.py \
  --data-samples-path assets/test_data_samples \
  --input-predictions-path assets/pred-test.csv \
  --output-perf-path assets/perf-test.json \
  --log-path assets/logs/test_metrics.log
```

You will need to update or follow the relevant path to add your assets or get the logs and model.

### Adding your dataset & the associated objective

You will now have to prepare the `opener.py` & `metrics.py` files, including their corresponding `description.md` and `Dockerfile` (where you have to add your dependencies). All assets are based on [substratools](https://github.com/SubstraFoundation/substra-tools/tree/master/substratools) classes to ease the process.

The `add_dataset_objective.py` script is in charge of registering your dataset as an asset of the `organisation` you belong to. With your dataset will also be registered the `opener.py` & `metrics.py` files. Before running the script, please make sure you edit it to update the following:

- `profile` in `client = substra.Client(profile_name="<PROFILE>")`, using the profile you created before
- `DATASET`: `name` & `permissions` (demo organisations are named `MyOrg1MSP` & `MyOrg2MSP`)
- `OBJECTIVE`: `name` & `permissions` (demo organisations are named `MyOrg1MSP` & `MyOrg2MSP`)
- any different path in regard to your project file structure

You can then run the command:

```sh
python3 add_dataset_objective.py
```

You will see some messages printed while the script is running. In the end, you should see a message like:

```sh
Assets keys have been saved to /home/user/your_project/assets_keys.json
```

This `assets_keys.json` file contains the keys of your registered assets that have been registered by the platform:

```json
{
  "dataset_key": "cb8d7f928d956ba6e9596a9b8624a0c7b0312eccd099f8791ca284bc05bd9416",
  "objective_key": "0a2e88018601d035f65788d2a081f84a35427474c4bde9fa357af2406be9b52c",
  "train_data_sample_keys": [
    "d72e2033aef1ce2f25cf56029a83a35bbc8b636b98d246b5931bc96ee5483f67",
    "6d2b0accd445a4cdd546f8af2d2eb9ae6b6462a6c4c00ae16ddb82500f6fa37e",
    "e72650db0206ec73edc74b5d1beab65bf487f6fb59c8a14563205602942cc287",
    "0827401963fa66beb2f2a74743b968aa7e213f1701c897317c4fa7e5b1d80193",
    "c624b9e66691542283333184972f909a621119001d159216aecbec10e0d91460",
    "34bf8fb5ecfa02656c8b61b314dc810b1de797ac849f5c23afbcaed5f2b1048a",
    "1dc200f6c7e22db5e725921c24af4127765afecca5e96f922a283f342a1a678a",
    "dde0b8762ea44c5821e460af167fc1e34659480cc3223891e0b2faecf23c5078",
    "9bd68627150f7c85bf46f20eefcb00d2f40cec64627e8f23d68e68d2e6da88c4",
    "22e81c5d24dd54a69909a265ff42c64afda93e2ff6bc3dc7225a3f3914c969bb"
  ],
  "test_data_sample_keys": [
    "fd4a10303ebbcc378679ff51ef592bd661f4923e306f69873cc7300c5e7c9f8c",
    "98af9156cd1445f8164d2b8b83f512d62ec67e8d2598a585a8564660c3aaef06"
  ],
}
```

Theses keys are useful when you want to get information about a specific resource, you can for example use:

```sh
substra get dataset cb8d7f928d956ba6e9596a9b8624a0c7b0312eccd099f8791ca284bc05bd9416 --profile node-1
```

### Adding your learning algorithm

You will need to update the `algo.py` file as well as the `description.md` & `Dockerfile`, and finally push it to the platform with the dedicated script:

```sh
python3 add_train_algo.py
```

Prior to running it, please make sure you edit it to update the following:

- `profile` in `client = substra.Client(profile_name="<PROFILE>")`, using the profile you created before
- `ALGO`: `name` &  `permissions` (demo organisations are named `MyOrg1MSP` & `MyOrg2MSP`)

Once done, the script will print some suggestions to gather information about your assets:

```sh
Assets keys have been saved to /home/user/your_project/assets_keys.json
Run the following commands to track the status of the tuples:
    substra get traintuple <traintuple_key> --profile node-1
    substra get testtuple <testtuple_key> --profile node-1
```

If you need to test that your algorithm is working without having access to the data, you can use
the `fake_data` from the `opener.py` by:

* **substra >= `0.7.0`**  
  editing the `add_train_algo.py` script and setup the client in debug mode by adding `debug=True`.

  ```python
  client = substra.Client(profile_name="<PROFILE>", debug=True)
  ```

* **substra < `0.7.0`**  
  using the CLI method `substra run-local`.

  ```sh
  substra run-local assets/algo \
    --train-opener=assets/dataset/opener.py \
    --test-opener=assets/dataset/opener.py \
    --metrics=assets/objective/ \
    --fake-data-samples
  ```

### Consult the performance from the ledger

Performances will be exported to a `perf.json` file. You can also use the CLI commands:

```sh
substra get traintuple <KEY> --profile <PROFILE>
substra get testtuple <KEY> --profile <PROFILE>
```

## 4. Resources

### Find help

Feeling lost? Remember, you can use `substra --help` anytime:

```sh
Usage: substra [OPTIONS] COMMAND [ARGS]...

  Substra Command Line Interface.

  For help using this tool, please open an issue on the Github
  repository: https://github.com/SubstraFoundation/substra

Options:
  --version    Show the version and exit.
  --help       Show this message and exit.

Commands:
  add          Add new asset to Substra platform.
  cancel       Cancel execution of an asset.
  config       Add profile to config file.
  describe     Display asset description.
  download     Download asset implementation.
  get          Get asset definition.
  leaderboard  Display objective leaderboard
  list         List assets.
  login        Login to the Substra platform.
  run-local    Run local.
  update       Update asset.
```

You can also get help for specific commands, for example `substra download --help`.

### Get inspired

The [Titanic example](https://github.com/SubstraFoundation/substra/blob/master/examples/titanic), included in the Substra repository, is ready to run but you can also have a look at the [community examples](https://github.com/SubstraFoundation/substra-examples):

- [Cross Validation](https://github.com/SubstraFoundation/substra/tree/master/examples/cross_val)
- [Compute plan](https://github.com/SubstraFoundation/substra/tree/master/examples/compute_plan)
- [Debugging](https://github.com/SubstraFoundation/substra/tree/master/examples/debugging)
- [MNIST](https://github.com/SubstraFoundation/substra-examples/tree/master/mnist)
- [MNIST with differential privacy](https://github.com/SubstraFoundation/substra-examples/tree/master/mnist-dp)
- [Deepfake detection](https://github.com/SubstraFoundation/substra-examples/tree/master/deepfake-detection)

### Documentation

- [General documentation](https://doc.substra.ai)
- [CLI documentation](https://github.com/SubstraFoundation/substra/blob/master/references/cli.md#summary)
- [Python SDK documentation](https://github.com/SubstraFoundation/substra/blob/master/references/sdk.md#substrasdk)
- [Substra cheat sheet](./cheatsheet_cli.md)