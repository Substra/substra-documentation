# Some common errors

TODO:

- permissions
- Docker/GPU checks
- CP

## CLI

[CLI documentation](https://github.com/SubstraFoundation/substra/blob/master/references/cli.md)

> Remember: you can use `substra --help` and `substra <command> --help` anytime!

### substra config

> `substra config --profile <profile> <url>`

### substra login

> `substra login --profile <profile> --username <username> --password '<password>'`

<details>
<summary><b>Requests error status 503</b></summary>

```sh
Requests error status 503: <html>
<head><title>503 Service Temporarily Unavailable</title></head>
    <body>
        <center><h1>503 Service Temporarily Unavailable</h1></center>
        <hr><center>openresty/1.15.8.2</center>
    </body>
</html>
Error: Request failed: HTTPError: 503 Server Error: Service Temporarily Unavailable for url: http://substra-backend.node-1.com/api-token-auth/
```

This issue is likely to be related to the server-side, and it indicates that the server is not responding (correctly). You can either try later or get in touch with the server administrator.
You can also try to reach the `/readiness` route to see if you get an `OK` answer, for example: `curl substra-backend.node-1.com/readiness`.
</details>

<details>
<summary><b>HTTPConnectionPool Error</b></summary>

```sh
Error: Request failed: ConnectionError: HTTPConnectionPool(host='<url>', port=8000): Max retries exceeded with url: /api-token-auth/ 
(Caused by NewConnectionError('<urllib3.connection.HTTPConnection object at 0x109f80c88>: Failed to establish a new connection: [Errno 61] Connection refused',))
```

This error occurs when you make more than X login calls in a given minute. You just need to wait a bit for it to disappear. 
The value of X depends on which value you set for the `DEFAULT_THROTTLE_RATES` environment variable. 
Default value when launching with skaffold is 120. 
(tips from [jmorel](https://github.com/jmorel) [here](https://github.com/SubstraFoundation/substra/issues/209))
</details>

### substra list/get/describe/download

### substra add/update

## Python SDK

[SDK documentation](https://github.com/SubstraFoundation/substra/blob/master/references/sdk.md)

> Remember: debug mode is awesome, use it! You just need to define your client like this `client = substra.Client(debug=True)`
> to be able to use `pdb`/`ipdb`!

### Generate data sample

Check the `generate_data_samples.py` script and the path to your dataset, the values of `N_TRAIN_DATA_SAMPLES` and `N_TEST_DATA_SAMPLES`.

<details>
<summary><b>FileExistsError</b></summary>
This indicates that the `train_data_samples` & `test_data_samples` have already been generated in the `assets` folder. This means that you will need to remove it before re-generating data samples.
</details>

### Register dataset and objective

Check the `add_dataset_objective.py` script and your assets (`DATASET`, `TEST_DATA_SAMPLES_PATHS`, `TRAIN_DATA_SAMPLES_PATHS`, `OBJECTIVE`, `METRICS_DOCKERFILE_FILES`)

### Add an algorithm

Check the `add_algo.py` script and your assets (`ALGO`, `ALGO_DOCKERFILE_FILES`).
