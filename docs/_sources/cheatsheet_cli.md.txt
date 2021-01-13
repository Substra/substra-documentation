# CLI Cheatsheet

## Preflight Checks

Use `substra --version` to ensure your installation is working and compatible with the server according to the [compatibility table](https://github.com/SubstraFoundation/substra#compatibility-table).

> Note: the `--force-reinstall` option might help if you need to install a new version of the package: `pip install substra==<VERSION> --force-reinstall`

You can then test that your substra CLI is able to reach the substra network with:

```sh
curl <DOMAIN_URL>/readiness

# Demo example
curl substra-backend.node-1.com/readiness
```

> Note: If you do not see the "OK" message, we might need to check your configuration, for example the `/etc/hosts` file.

## Substra config

You can setup a profile, this is useful if you work with several configurations, but not required, so you can omit the `--profile <PROFILE_NAME>` option if you don't use a specific profile.

```sh
# Usage
substra config [OPTIONS] <URL>

# With a profile
substra config --profile <PROFILE_NAME> <URL>

# Demo example
substra config --profile node-1 http://substra-backend.node-1.com
```

> Note: the profile name is a local resource located in `~/.substra`, this means that you can use any name you want.

## Substra login

```sh
# Usage
substra login [OPTIONS]

# You can simply use
substra login

# Or provide the required parameter like this
substra login --profile <PROFILE_NAME> --username <USERNAME> --password <PASSWORD>

# Demo example
substra login --profile node-1 --username node-1 --password 'p@$swr0d44'
```

There are also a bunch of useful options like `--log-level DEBUG` or `--verbose`.

## List & Get assets

Assets you can list:

- `node`: nodes in substra network
- `dataset`: registered datasets
- `objective`: Machine Learning question
- `algo`: registered algorithm
- `compute_plan`: the "blueprint" of tasks to be executed accross the substra network

```sh
substra list node # with the profile: substra list node --profile node-1
substra list dataset
substra list objective
substra list algo
substra list complute_plan

substra list traintuple
substra list testtupuble
```

Theses commands will return the **keys** of the assets that you will then need to use commands like `get`, `describe` or `download`.

You will then have to use:

```sh
substra get traintuple <KEY>
substra get testtuple <KEY>
substra get compute_plan <KEY>
substra get algo <KEY>
```

For example, you will get more information about this `testuple`, like its `STATUS`, `RANK`, `PERF` or associated `PERMISSIONS`:

```sh
substra get testtuple d8238915246f02c3b0f8cd4aa0a9546b31daa72477f98fddb5b2137a353d0c1c
KEY                         d8238915246f02c3b0f8cd4aa0a9546b31daa72477f98fddb5b2137a353d0c1c
TRAINTUPLE KEY              28d4c395daac2aa80f46792cbeb4572135cd0aa06042b47f400de65271eacc2f
TRAINTUPLE TYPE             traintuple
ALGO KEY                    7e039d490df04ed3f613191d3af004a1147f22b1f28b942f11a6c830caeb9e9b
ALGO NAME                   Titanic: Random Forest
OBJECTIVE KEY               1158d2f5c0cf9f80155704ca0faa28823b145b42ebdba2ca38bd726a1377e1cb
CERTIFIED                   True
STATUS                      doing
PERF                        0
DATASET KEY                 c1d9b42d9538f825109c38c4f599d47391347d198a0770331ba790b7ebcfaa40
TEST DATA SAMPLE KEYS       2 keys
RANK                        0
TAG
COMPUTE PLAN ID
LOG
CREATOR                     MyOrg1MSP
WORKER                      MyOrg1MSP
PERMISSIONS                 Processable by its owner only
```

## Register assets
 
### Dataset

```sh
substra add dataset [OPTIONS] PATH
substra add dataset --help
```

### Data samples

```sh
substra add data_sample [OPTIONS] PATH
substra add data_sample --help
```

You can use the same pattern for registering several kind of assets: `objective`, `compute_plan`, `algo`, `testtuple`, `traintuple`, etc.

## Tips

You can use `--output` formatting options like `-o yaml` or `-o json`; `pretty` is used by default.

## Help

Remember, you can use anytime the substra CLI `--help` command which will provide you with a lot of resources for each asset and method!

```sh
substra --help
substra <command> --help
```

## Resources

- [General documentation](https://doc.substra.ai/)
- [CLI documentation](https://github.com/SubstraFoundation/substra/blob/master/references/cli.md)
- [Python SDK documentation](https://github.com/SubstraFoundation/substra/blob/master/references/sdk.md)
- [Titanic examples](https://github.com/SubstraFoundation/substra/blob/master/examples/titanic)
- [Community examples](https://github.com/SubstraFoundation/substra-examples)
- [Slack #help channel](https://substra-workspace.slack.com/archives/CT54J1U2E)
