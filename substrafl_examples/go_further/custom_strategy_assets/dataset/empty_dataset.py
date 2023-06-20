from pathlib import Path

from substra.sdk.schemas import DatasetSpec
from substra.sdk.schemas import Permissions
from substra.sdk.schemas import DataSampleSpec
from substrafl.nodes import TrainDataNode, AggregationNode, TestDataNode


def setup_empty_dataset(clients, algo_org_id, data_provider_id, metric_function):
    # Create the temporary directory for generated data
    empty_data_path = Path.cwd() / "tmp"
    empty_data_path.mkdir(exist_ok=True)

    permissions_dataset = Permissions(public=False, authorized_ids=[algo_org_id])

    dataset = DatasetSpec(
        name="Custom",
        type="empty",
        data_opener=Path.cwd() / "custom_strategy_assets" / "dataset" / "empty_opener.py",
        description=Path.cwd() / "custom_strategy_assets" / "dataset" / "description.md",
        permissions=permissions_dataset,
        logs_permission=permissions_dataset,
    )

    dataset_keys = {}
    train_datasample_keys = {}
    test_datasample_keys = {}
    train_data_nodes = []
    test_data_nodes = []
    for org_id in data_provider_id:
        client = clients[org_id]

        # Add the dataset to the client to provide access to the opener in each organization.
        dataset_keys[org_id] = client.add_dataset(dataset)
        assert dataset_keys[org_id], "Missing data manager key"

        client = clients[org_id]

        # Add the training data on each organization.
        data_sample = DataSampleSpec(
            data_manager_keys=[dataset_keys[org_id]],
            path=empty_data_path,
        )
        train_datasample_keys[org_id] = client.add_data_sample(
            data_sample,
            local=True,
        )

        # Add the testing data on each organization.
        data_sample = DataSampleSpec(
            data_manager_keys=[dataset_keys[org_id]],
            path=empty_data_path,
        )
        test_datasample_keys[org_id] = client.add_data_sample(
            data_sample,
            local=True,
        )

        train_data_nodes.append(
            TrainDataNode(
                organization_id=org_id,
                data_manager_key=dataset_keys[org_id],
                data_sample_keys=[train_datasample_keys[org_id]],
            )
        )

        test_data_nodes.append(
            TestDataNode(
                organization_id=org_id,
                data_manager_key=dataset_keys[org_id],
                test_data_sample_keys=[test_datasample_keys[org_id]],
                metric_functions=metric_function,
            )
        )

    aggregation_node = AggregationNode(algo_org_id)

    # Create the Train Data Nodes (or training tasks) and save them in a list

    return train_data_nodes, test_data_nodes, aggregation_node
