import argparse
import subprocess
import sys
from pathlib import Path


def arg_parse():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--write-summary-to-file",
        type=str,
        default=None,
        help="Write a summary of the results to the given filename",
    )
    parser.add_argument(
        "--docker-mode",
        type=bool,
        default=False,
        help="Run the examples in docker mode",
    )
    args = parser.parse_args()
    return args


def write_summary_file(test_passed: bool, test_name: str, path: Path):
    with path.open("w") as fp:
        if test_passed is not None:
            res = "✅" if test_passed else "❌"
        else:
            res = "⏭ (skipped)"
        fp.write(f"{res} {test_name} - python version {sys.version_info.major}.{sys.version_info.minor} \n")


def run_titanic(summary_file: str, docker_mode=False):
    test_name = "Substra titanic example"
    example_path = "examples/titanic_example"
    cmd = ""
    try:
        if docker_mode:
            cmd += "export SUBSTRA_FORCE_EDITABLE_MODE=True; "
            cmd += "export SUBSTRA_ORG_1_BACKEND_TYPE=docker; "

        cmd += f"pip install -r {example_path}/assets/requirements.txt; "
        cmd += f"python {example_path}/run_titanic.py;"
        subprocess.check_call([cmd], shell=True)
        test_passed = True
        return test_passed

    except subprocess.CalledProcessError:
        print(f"FATAL: {test_name} completed with a non-zero exit code.")
        test_passed = False
        return test_passed

    finally:
        if summary_file:
            write_summary_file(test_passed=test_passed, test_name=test_name, path=Path(summary_file))


def run_diabetes(summary_file: str, docker_mode=False):
    test_name = "Substra diabetes example"
    example_path = "examples/diabetes_example"
    cmd = ""
    try:
        if docker_mode:
            cmd += "export SUBSTRA_FORCE_EDITABLE_MODE=True; "
            cmd += "export SUBSTRA_ORG_1_BACKEND_TYPE=docker; "
            cmd += "export SUBSTRA_ORG_2_BACKEND_TYPE=docker; "
            cmd += "export SUBSTRA_ORG_3_BACKEND_TYPE=docker; "

        cmd += f"pip install -r {example_path}/assets/requirements.txt; "
        cmd += f"python {example_path}/run_diabetes.py;"
        subprocess.check_call([cmd], shell=True)
        test_passed = True
        return test_passed

    except subprocess.CalledProcessError:
        print(f"FATAL: {test_name} completed with a non-zero exit code.")
        test_passed = False
        return test_passed

    finally:
        if summary_file:
            write_summary_file(test_passed=test_passed, test_name=test_name, path=Path(summary_file))


def run_mnist(summary_file: str, docker_mode=False):
    test_name = "MNIST example"
    example_path = "substrafl_examples/get_started"
    cmd = ""
    try:
        if docker_mode:
            cmd += "export SUBSTRA_FORCE_EDITABLE_MODE=True; "
            cmd += "export SUBSTRA_ORG_1_BACKEND_TYPE=docker; "
            cmd += "export SUBSTRA_ORG_2_BACKEND_TYPE=docker; "
            cmd += "export SUBSTRA_ORG_3_BACKEND_TYPE=docker; "

        cmd += f"pip install -r {example_path}/assets/requirements.txt; "
        cmd += f"python {example_path}/run_titanic.py;"
        subprocess.check_call([cmd], shell=True)
        test_passed = True
        return test_passed

    except subprocess.CalledProcessError:
        print("FATAL: `Titanic example` completed with a non-zero exit code.")
        test_passed = False
        return test_passed

    finally:
        if summary_file:
            write_summary_file(test_passed=test_passed, test_name=test_name, path=Path(summary_file))


def run_iris(summary_file: str, docker_mode=False):
    test_name = "SubstraFL IRIS example"
    example_path = "substrafl_examples/go_further"
    cmd = ""
    try:
        if docker_mode:
            cmd += "export SUBSTRA_FORCE_EDITABLE_MODE=True; "
            cmd += "export SUBSTRA_ORG_1_BACKEND_TYPE=docker; "
            cmd += "export SUBSTRA_ORG_2_BACKEND_TYPE=docker; "
            cmd += "export SUBSTRA_ORG_3_BACKEND_TYPE=docker; "

        cmd += f"pip install -r {example_path}/assets/requirements.txt; "
        cmd += f"python {example_path}/run_titanic.py;"
        subprocess.check_call([cmd], shell=True)
        test_passed = True
        return test_passed

    except subprocess.CalledProcessError:
        print(f"FATAL: {test_name} completed with a non-zero exit code.")
        test_passed = False
        return test_passed

    finally:
        if summary_file:
            write_summary_file(test_passed=test_passed, test_name=test_name, path=Path(summary_file))


def main():
    args = arg_parse()

    success = True

    success = run_titanic(summary_file=args.write_summary_to_file, docker_mode=args.docker_mode) and success
    success = run_diabetes(summary_file=args.write_summary_to_file, docker_mode=args.docker_mode) and success
    success = run_mnist(summary_file=args.write_summary_to_file, docker_mode=args.docker_mode) and success
    success = run_iris(summary_file=args.write_summary_to_file, docker_mode=args.docker_mode) and success

    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
