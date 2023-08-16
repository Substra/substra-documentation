
examples: example-substra example-substrafl

example-substra: example-diabete example-titanic
example-diabete:
	cd examples/diabetes_example/ && python run_diabetes.py
example-titanic:
	cd examples/titanic_example/ && python run_titanic.py

example-substrafl: example-mnist example-iris
example-mnist:
	cd substrafl_examples/get_started/ && python run_mnist_torch.py
example-iris:
	cd substrafl_examples/go_further/ && python run_iris_sklearn.py