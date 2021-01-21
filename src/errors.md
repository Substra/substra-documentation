# Some common errors

TODO:

- permissions
- Docker/GPU checks
- CP

## CLI

CLI documentation: <https://github.com/SubstraFoundation/substra/blob/master/references/cli.md>

> Remember: you can use `substra --help` and `substra <command> --help` anytime!

### substra config

### substra login

If you try to login with substra cli (`substra login --profile <profile> --username <username> --password '<password>'
`) and get a `Requests error status 503`, followed by:

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

This issue is likely to be related to the server-side, and it indicates that the server is not responding correctly. You can either try later or get in touch with the server administrator.
You can try to reach the `/readiness` route to see if you get an `OK` answer, for example: `curl substra-backend.node-1.com/readiness`.

### substra list/get/describe/download

### substra add/update

## Python SDK

SDK documentation: <https://github.com/SubstraFoundation/substra/blob/master/references/sdk.md>

> Remember: debug mode is awesome, use it! You just need to define your client like this `client = substra.Client(debug=True)`
> to be able to use `pdb`/`ipdb`!

### Generate data sample

Check the `generate_data_samples.py` script and the path to your dataset, the values of `N_TRAIN_DATA_SAMPLES` and `N_TEST_DATA_SAMPLES`.

- `FileExistsError` indicates that the `train_data_samples` & `test_data_samples` have already been generated in the `assets` folder. This means that you will need to remove it before re-generating data samples.

### Register dataset and objective

Check the `add_dataset_objective.py` script and your assets (`DATASET`, `TEST_DATA_SAMPLES_PATHS`, `TRAIN_DATA_SAMPLES_PATHS`, `OBJECTIVE`, `METRICS_DOCKERFILE_FILES`)

### Add an algorithm

Check the `add_algo.py` script and your assets (`ALGO`, `ALGO_DOCKERFILE_FILES`).
