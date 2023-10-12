install-examples-dependencies:
	pip3 install -r examples_requirements.txt

examples: examples-substra examples-substrafl

examples-substra: example-core-diabetes example-core-titanic

example-core-diabetes:
	cd docs/source/examples/substra_core/diabetes_example/ && ipython -c "%run run_diabetes.ipynb"
example-core-titanic:
	cd docs/source/examples/substra_core/titanic_example/ && ipython -c "%run run_titanic.ipynb"

examples-substrafl: example-fl-mnist example-fl-iris example-fl-cyclic example-fl-diabetes

example-fl-mnist:
	cd docs/source/examples/substrafl/get_started/ && ipython -c "%run run_mnist_torch.ipynb"
example-fl-iris:
	cd docs/source/examples/substrafl/go_further/ && ipython -c "%run run_iris_sklearn.ipynb"
example-fl-cyclic:
	cd docs/source/examples/substrafl/go_further/ && ipython -c "%run run_mnist_cyclic.ipynb"
example-fl-diabetes:
	cd docs/source/examples/substrafl/go_further/ && ipython -c "%run run_diabetes_substrafl.ipynb"