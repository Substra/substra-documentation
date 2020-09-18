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

> Note: Substra CLI isn't compatible yet with Windows unless you use the Linux Sub System. Please have a look at those resources:
>
> - WSL: <https://docs.microsoft.com/en-us/windows/wsl/install-win10>
> - Vscode: <https://code.visualstudio.com/docs/remote/wsl>
> - Docker: <https://docs.docker.com/docker-for-windows/wsl/>

From this point onwards, substra command line interface will have autocompletion enabled.

## Documentation

Interacting with the Substra platform

- [CLI](https://github.com/SubstraFoundation/substra/blob/master/references/cli.md#summary)
- [SDK](https://github.com/SubstraFoundation/substra/blob/master/references/sdk.md#substrasdk)

To help you implement your assets in Python, please have a look to the [Substra Tools](https://github.com/SubstraFoundation/substra-tools).

## Examples

Substra comes with a set of handy [examples](https://github.com/SubstraFoundation/substra/blob/master/examples) that will help you getting started! If you want to go further, you can also have a look at the [community driven examples](https://github.com/SubstraFoundation/substra-examples) where you will see how to implement MNIST, MNIST with differential privacy or Deepfake detection!

