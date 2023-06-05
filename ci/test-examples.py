import argparse
import subprocess
import sys
from pathlib import Path
import os


def arg_parse():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--write-summary-to-file",
        type=str,
        default=None,
        help="Write a summary of the results to the given filename",
    )
    args = parser.parse_args()
    return args


def write_summary_file(test_passed: bool, test_name: str, path: Path):
    with path.open("a") as fp:
        if test_passed is not None:
            res = "✅" if test_passed else "❌"
        else:
            res = "⏭ (skipped)"
        fp.write(f"{res} {test_name} - python version {sys.version_info.major}.{sys.version_info.minor} \n")


def run_example(
    name: str,
    example_file: Path,
    requirements_relative_path: Path,
    summary_file: Path,
):
    cmd = ""
    test_passed = None

    try:
        cmd += f"pip install -r {requirements_relative_path}; "
        cmd += f"python {example_file.name};"
        subprocess.check_call([cmd], cwd=example_file.parent, shell=True, env=dict(os.environ))
        test_passed = True
        return test_passed

    except subprocess.CalledProcessError:
        print(f"FATAL: {name} completed with a non-zero exit code.")
        test_passed = False
        return test_passed

    finally:
        if summary_file:
            write_summary_file(test_passed=test_passed, test_name=name, path=summary_file)


def main():
    args = arg_parse()

    success = True

    success = (
        run_example(
            name="Substra Titanic example",
            example_file=Path("examples") / "titanic_example" / "run_titanic.py",
            requirements_relative_path=Path("assets") / "requirements.txt",
            summary_file=Path(args.write_summary_to_file),
        )
        and success
    )
    success = (
        run_example(
            name="Substra Diabetes example",
            example_file=Path("examples") / "diabetes_example" / "run_diabetes.py",
            requirements_relative_path=Path("assets") / "requirements.txt",
            summary_file=Path(args.write_summary_to_file),
        )
        and success
    )
    success = (
        run_example(
            name="SubstraFL MNIST example",
            example_file=Path("substrafl_examples") / "get_started" / "run_mnist_torch.py",
            requirements_relative_path=Path("torch_fedavg_assets") / "requirements.txt",
            summary_file=Path(args.write_summary_to_file),
        )
        and success
    )
    success = (
        run_example(
            name="SubstraFL IRIS example",
            example_file=Path("substrafl_examples") / "go_further" / "run_iris_sklearn.py",
            requirements_relative_path=Path("sklearn_fedavg_assets") / "requirements.txt",
            summary_file=Path(args.write_summary_to_file),
        )
        and success
    )

    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
