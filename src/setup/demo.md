# Hands on Substra demo

Check this [page](https://www.substra.ai/en/demo) and get in touch with us, we will then provide you with instructions and credentials!

## 1. Configuration

### Install the Substra client

> Substra is not yet compatible with windows, still you can manage to make it work the Windows Linux Subsystem, see the [documentation](https://doc.substra.ai/usage.html?highlight=wsl#installation) or inside a Linux virtual machine.

It will install the [CLI](https://github.com/SubstraFoundation/substra/blob/master/references/cli.md#summary) (interactive command line interface) & the [Python SDK](https://github.com/SubstraFoundation/substra/blob/master/references/sdk.md#substrasdk) (ready to be used in your Python scripts). You can also clone the [Substra repository](https://github.com/SubstraFoundation/substra) to get the full examples located in the `example` folder.

```sh
pip install substra
# Or if a specific version is required by the team
pip install substra==0.6.0 --force-reinstall

# Check that Substra is working
substra --version
# For example: substra, version 0.6.0
```

You will also need to install the [Substra Tools](https://github.com/SubstraFoundation/substra-tools/tree/master/substratools) package which contains all assets templates you will need to submit your algo to the Substra network:

```sh
pip install substratools
```

### Edit your hosts file

As the demo instance is a demo instance, you will have to create a link between the server ip address (provided with your credentials) and the 2 nodes domains used in the demo:

```sh
# Replace <CHANGE_ME> with the ip provided by the team
echo "<CHANGE_ME> substra-backend.node-1.com substra-frontend.node-1.com substra-backend.node-2.com substra-frontend.node-2.com" | sudo tee -a /etc/hosts
```

This will add a line like the following to the file `/etc/hosts` of your machine:

```sh
# 01.02.003.004     substra-backend.node-1.com substra-backend.node-2.com substra-frontend.node-1.com substra-frontend.node-2.com
```

In order to check that your local substra client can reach the demo server, you can use the following command:

```sh
# Quick test on the backend of the node-1
curl substra-backend.node-1.com/readiness
```

You should receive an `"OK"` answer, if not please check again your `/etc/hosts` file. You can also use your browser to visit [substra-frontend.node-1.com](http:substra-frontend.node-1.com) & [substra-frontend.node-2.com](http:substra-frontend.node-2.com) (without https, because this is demo).

### Login

You can now configure your local Substra client, *with the credentials you received from us*, to connect to the `node-1` which hosts the organisation `org-1`:

```sh
# Configure the client, with a profile, a user, a password and an url
substra config --profile <PROFILE> -u <USER> -p "<PASSWORD>" http://substra-backend.node-1.com

# Login with the profile you created (located in your home folder: ~/.substra)
substra login --profile <PROFILE>

# You can then list the network nodes that are on the Substra network
substra list node --profile <PROFILE>

# And list, for example, the registered datasets in the node you are logged in
substra list dataset --profile <PROFILE>
```

## 2. Follow Titanic example

You will find everything you need to run a first end to end example with the tabular data of the Titanic: [repository](https://github.com/SubstraFoundation/substra/blob/master/examples/titanic/README.md).

Please note that each resource is referenced by its hash in Substra, a unique identifier, so if you want to try out the ready to use Titanic example, you will need to edit, at least, the python files and the `description.md` for the assets (like adding a `print()` statement and editing the `description.md` files).

## 3. Work with your own assets

> **Note**: This is demo instance, please do not use any sensitive asset on this platform!

The special [debugging section](https://github.com/SubstraFoundation/substra/tree/master/examples/debugging) can be off help preparing your asset, please have a look!

### Data preparation

The first step is partitioning your data, and create a test dataset and a train dataset. Following the Titanic example, you will first need to generate data samples (in `test_data_samples` and `train_data_samples` folders) by updating the `generate_data_samples` script with the path to you dataset and running it:

```sh
pip install -r scripts/requirements.txt
python scripts/generate_data_samples.py
```

### Add dataset & objective

You will now have to prepare the `opener.py` & `metrics.py` files, including their corresponding `description.md` and `Dockerfile` (where you have to add your depedencies). All assets are based on [substratools](https://github.com/SubstraFoundation/substra-tools/tree/master/substratools) classes to ease the process.

The `add_dataset_objective.py` script is in charge of registering data to the `organisation`.

Before running the script, please make sure you have updated:

- `profile` in `client = substra.Client(profile_name="<PROFILE>")`, using the profile you created before
- `DATASET`: `name` & `permissions` (demo organisations are named `MyOrg1MSP` & `MyOrg2MSP`)
- `OBJECTIVE`: `name` & `permissions` (demo organisations are named `MyOrg1MSP` & `MyOrg2MSP`)
- any different path in regard to project file structure

You can then run the command:

```sh
python3 add_dataset_objective.py
```

### Add your algorithm

You will need to update the `algo.py` file as well as the `description.md` & `Dockerfile`, and finally push it to the platform with the dedicated script:

```sh
python3 add_train_algo.py
```

In the `algo.py` file, please make sure you have updated:

- `profile` in `client = substra.Client(profile_name="<PROFILE>")`, using the profile you created before
- `ALGO`: `name` &  `permissions` (demo organisations are named `MyOrg1MSP` & `MyOrg2MSP`)

If you need to test that your algorithm is working without having access to the data, you can use the CLI method `substra run-local` that will use the `fake_data` method from the `opener.py`:

```sh
substra run-local assets/algo \
  --train-opener=assets/dataset/opener.py \
  --test-opener=assets/dataset/opener.py \
  --metrics=assets/objective/ \
  --fake-data-samples
```

### Training & Testing tasks

```sh
# Train
python3 assets/algo/algo.py train \
  --debug \
  --opener-path assets/dataset/opener.py \
  --data-samples-path assets/train_data_samples \
  --output-model-path assets/model/model \
  --log-path assets/logs/train.log

# Predict
python assets/algo/algo.py predict \
  --debug \
  --opener-path assets/dataset/opener.py \
  --data-samples-path assets/train_data_samples \
  --output-predictions-path assets/pred-train.csv \
  --models-path assets/model/ \
  --log-path assets/logs/train_predict.log \
  model

# Performance
python assets/objective/metrics.py \
  --debug \
  --opener-path assets/dataset/opener.py \
  --data-samples-path assets/train_data_samples \
  --input-predictions-path assets/pred-train.csv \
  --output-perf-path assets/perf-train.json \
  --log-path assets/logs/train_metrics.log

# Test / predict
python assets/algo/algo.py predict \
  --debug \
  --opener-path assets/dataset/opener.py \
  --data-samples-path assets/test_data_samples \
  --output-predictions-path assets/pred-test.csv \
  --models-path assets/model/ \
  --log-path assets/logs/test_predict.log \
  model

# Test / performance
python assets/objective/metrics.py \
  --debug \
  --opener-path assets/dataset/opener.py \
  --data-samples-path assets/test_data_samples \
  --input-predictions-path assets/pred-test.csv \
  --output-perf-path assets/perf-test.json \
  --log-path assets/logs/test_metrics.log
```

You will need to update or follow the relevant path to add your assets or get the logs and model.

## 4. Resources

### Find help

Feeling lost? Remember, you can use `substra --help` anytime:

```sh
Usage: substra [OPTIONS] COMMAND [ARGS]...

  Substra Command Line Interface.

  For help using this tool, please open an issue on the Github
  repository: https://github.com/SubstraFoundation/substra

Options:
  --version  Show the version and exit.
  --help     Show this message and exit.

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

The [Titanic example](https://github.com/SubstraFoundation/substra/blob/master/examples/titanic/README.md), included in the Substra repository, is ready to run but you can also have a look at the [community examples](https://github.com/SubstraFoundation/substra-examples):

- [Titanic](https://github.com/SubstraFoundation/substra/tree/master/examples/titanic)
- [Cross Validation](https://github.com/SubstraFoundation/substra/tree/master/examples/cross_val)
- [Compute plan](https://github.com/SubstraFoundation/substra/tree/master/examples/compute_plan)
- [Debugging](https://github.com/SubstraFoundation/substra/tree/master/examples/debugging)
- [MNIST](https://github.com/SubstraFoundation/substra-examples/tree/master/mnist)
- [MNIST with Tensorflow Privacy](https://github.com/SubstraFoundation/substra-examples/tree/master/mnist-dp)
- [Deepfake detection](https://github.com/SubstraFoundation/substra-examples/tree/master/deepfake-detection)

### Documentation

- [General documentation](https://doc.substra.ai)
- [CLI documentation](https://github.com/SubstraFoundation/substra/blob/master/references/cli.md#summary)
- [Python SDK documentation](https://github.com/SubstraFoundation/substra/blob/master/references/sdk.md#substrasdk)
