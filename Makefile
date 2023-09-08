install-examples-dependencies:
	pip3 install -r examples/substra_core/diabetes_example/assets/requirements.txt \
    -r examples/substra_core/titanic_example/assets/requirements.txt \
    -r examples/substrafl/get_started/torch_fedavg_assets/requirements.txt \
    -r examples/substrafl/go_further/sklearn_fedavg_assets/requirements.txt \
    -r examples/substrafl/go_further/torch_cyclic_assets/requirements.txt \
    -r examples/substrafl/go_further/diabetes_substrafl_assets/requirements.txt \

examples: example-substra example-substrafl

example-substra: example-core-diabetes example-core-titanic

example-core-diabetes:
	cd examples/substra_core/diabetes_example/ && python run_diabetes.py
example-core-titanic:
	cd examples/substra_core/titanic_example/ && python run_titanic.py

example-substrafl: example-fl-mnist example-fl-iris example-fl-cyclic example-fl-diabetes

example-fl-mnist:
	cd examples/substrafl/get_started/ && python run_mnist_torch.py
example-fl-iris:
	cd examples/substrafl/go_further/ && python run_iris_sklearn.py
example-fl-cyclic:
	cd examples/substrafl/go_further/ && python run_iris_sklearn.py
example-fl-diabetes:
	cd examples/substrafl/go_further/ && python run_iris_sklearn.py