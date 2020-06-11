# Usage

CLI and SDK for interacting with Substra platform.

## Installation

To install the command line interface and the python sdk, run the following command:

```sh
pip install substra
```

To enable Bash completion, you need to put into your `.bashrc`:

```sh
eval "$(_SUBSTRA_COMPLETE=source substra)"
```

For zsh users add this to your .zshrc:

```sh
eval "$(_SUBSTRA_COMPLETE=source_zsh substra)"
```

From this point onwards, substra command line interface will have autocompletion enabled.

## Documentation

Interacting with the Substra platform

- [Command line interface](https://github.com/SubstraFoundation/substra/blob/master/references/cli.md#summary)
- [SDK](https://github.com/SubstraFoundation/substra/blob/master/references/sdk.md#substrasdk)

Implementing your assets in Python with [Substra Tools](https://github.com/SubstraFoundation/substra-tools)

- [Objective base class](https://github.com/SubstraFoundation/substra-tools/blob/master/docs/api.md#metrics)
- [Dataset base class](https://github.com/SubstraFoundation/substra-tools/blob/master/docs/api.md#opener)
- [Algo base class](https://github.com/SubstraFoundation/substra-tools/blob/master/docs/api.md#algo)
- [Composite algo base class](https://github.com/SubstraFoundation/substra-tools/blob/master/docs/api.md#compositealgo)
- [Aggregate algo base class](https://github.com/SubstraFoundation/substra-tools/blob/master/docs/api.md#aggregatealgo)

Learning about the Substra platform

- [Concepts](https://github.com/SubstraFoundation/substra/blob/master/docs/concepts.md#substras-concepts)
- [Machine Learning tasks](https://github.com/SubstraFoundation/substra/blob/master/docs/ml_tasks.md#machine-learning-tasks)
- [Adding a full pipeline](https://github.com/SubstraFoundation/substra/blob/master/docs/full_pipeline_workflow.md#adding-a-full-pipeline)
- [Adding data samples](https://github.com/SubstraFoundation/substra/blob/master/docs/add_data_samples.md#add-data-samples-to-substra)

## Examples

- [Titanic](https://github.com/SubstraFoundation/substra/blob/master/examples/titanic/README.md#titanic)
- [Cross-validation](https://github.com/SubstraFoundation/substra/blob/master/examples/cross_val/README.md#cross-validation)
- [Compute plan](https://github.com/SubstraFoundation/substra/blob/master/examples/compute_plan/README.md#compute-plan)
