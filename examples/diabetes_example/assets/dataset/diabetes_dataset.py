from sklearn.datasets import load_diabetes
import pandas as pd
import pathlib

def setup_diabetes(data_path: pathlib.Path):
    raw_data = load_diabetes(scaled=False)

    description_file = data_path / "description.md"
    description_file.touch()
    description_file.write_text(raw_data.DESCR)

    dataset = pd.DataFrame(data=raw_data.data, columns=raw_data.feature_names)
    # map the "sex" column to categorical data
    dataset["sex"] = dataset["sex"].replace({1: 'M', 2: 'F'}).astype("category")

    # Create folders for both organisations
    (data_path / "org_1").mkdir(exist_ok=True)
    (data_path / "org_2").mkdir(exist_ok=True)

    # Split the dataset in two uneven parts
    split_index = int(len(dataset) * 2/3)
    dataset.iloc[:split_index].to_csv(data_path/ "org_1" /"data.csv", index=False)
    dataset.iloc[split_index:].to_csv(data_path/ "org_2"  /"data.csv", index=False)
