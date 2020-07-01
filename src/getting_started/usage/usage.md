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

To help you implement your assets in Python, please have a look to the [Substra Tools](https://github.com/SubstraFoundation/substra-tools).

## Examples

- [Titanic](https://github.com/SubstraFoundation/substra/blob/master/examples/titanic/README.md#titanic)
- [Cross-validation](https://github.com/SubstraFoundation/substra/blob/master/examples/cross_val/README.md#cross-validation)
- [Compute plan](https://github.com/SubstraFoundation/substra/blob/master/examples/compute_plan/README.md#compute-plan)
