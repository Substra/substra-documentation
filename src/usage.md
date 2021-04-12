# Client

CLI and SDK for interacting with Substra platform.

## Installation

To install the command line interface and the Python SDK, run the following command:

```sh
pip install substra
```

To enable Bash completion, you need to put into your `.bashrc`:

```sh
eval "$(_SUBSTRA_COMPLETE=source substra)"
```

For zsh users add this to your `.zshrc`:

```sh
eval "$(_SUBSTRA_COMPLETE=source_zsh substra)"
```

> Note: Substra CLI isn't compatible yet with Windows unless you use the Linux Sub System. Please have a look at those resources:
>
> - WSL: <https://docs.microsoft.com/en-us/windows/wsl/install-win10>
> - Vscode: <https://code.visualstudio.com/docs/remote/wsl>
> - Docker: <https://docs.docker.com/docker-for-windows/wsl/>

From this point onward, substra command line interface will have autocompletion enabled.

## Documentation

Interacting with the Substra platform

- [CLI](https://github.com/SubstraFoundation/substra/blob/master/references/cli.md#summary)
- [SDK](https://github.com/SubstraFoundation/substra/blob/master/references/sdk.md#substrasdk)

To help you implement your assets in Python, please have a look to the [Substra Tools](https://github.com/SubstraFoundation/substra-tools).

## Cheat sheet

You can use this general [cheat sheet](./cheatsheet_cli.md) to help you getting started with Substra.

## Permissions

Learn more about asset Permissions in [this document](./permissions.md).

## Common errors

If you are facing some issues manipulating your assets, we have a dedicated page [here](./errors.md).

## Examples

Substra comes with a set of handy [examples](https://github.com/SubstraFoundation/substra/blob/master/examples) that will help you getting started! If you want to go further, you can also have a look at the [community driven examples](https://github.com/SubstraFoundation/substra-examples) where you will see how to implement MNIST, MNIST with differential privacy or Deepfake detection!

## Hands on Substra

If you are facing issues with Substra (CLI or SDK), you can have a look at:

- Github [issues](https://github.com/SubstraFoundation/substra/issues)
- [Debugging](https://doc.substra.ai/debugging.html)
- Join our [Slack](https://substra.us18.list-manage.com/track/click?e=2effed55c9&id=fa49875322&u=385fa3f9736ea94a1fcca969f)
