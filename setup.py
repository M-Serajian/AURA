#!/usr/bin/env python3
import os
import sys
import subprocess
import platform
import random
import string
import argparse
from shutil import rmtree

# --- Fix potential environment corruption ---
for var in ("PYTHONPATH", "PYTHONHOME"):
    os.environ.pop(var, None)

# --- CLI color codes ---
COLORS = {
    "RED": "\033[91m",
    "GREEN": "\033[92m",
    "YELLOW": "\033[93m",
    "NC": "\033[0m",  # No color
}

def print_colored(message, color="NC"):
    if platform.system() == "Windows" and not os.environ.get("TERM"):
        print(message)
    else:
        print(f"{COLORS[color]}{message}{COLORS['NC']}")

def run_command(command, shell=False, cwd=None):
    try:
        result = subprocess.run(command, shell=shell, check=True, cwd=cwd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        print_colored(result.stdout.decode(), "GREEN")
        return True
    except subprocess.CalledProcessError as e:
        print_colored(e.stderr.decode(), "RED")
        return False

def generate_env_name(prefix="resistance-predictor-env", length=6):
    return f"{prefix}-{''.join(random.choices(string.ascii_lowercase + string.digits, k=length))}"

def create_environment(env_name):
    print_colored(f"[INFO] Creating environment '{env_name}'...", "YELLOW")
    return run_command(["conda", "create", "-n", env_name, "python=3.10.8", "-y"])

def install_dependencies(env_name):
    packages = {
        "xgboost": "3.0.2",  # Latest version as of July 2025
        "scipy": "1.11.4",
        "gcc_linux-64": "12.2.0",
        "gxx_linux-64": "12.2.0",
        "boost-cpp": "1.77.0",
        "cmake": "3.26.4",
        "git": "2.40.1",
        "pandas": "",       # latest version
        "pyarrow": "",      # for reading .parquet
    }

    for package, version in packages.items():
        pkg_str = f"{package}={version}" if version else package
        print_colored(f"[INFO] Installing {pkg_str}...", "YELLOW")
        run_command(["conda", "install", "-n", env_name, pkg_str, "-c", "conda-forge", "-y"])

    print_colored("[INFO] Installing pip package: simplejson...", "YELLOW")
    run_command(["conda", "run", "-n", env_name, "pip", "install", "simplejson"])

def setup_and_compile_kmx(env_name):
    base_dir = os.getcwd()
    include_dir = os.path.join(base_dir, "include")
    kmx_dir = os.path.join(include_dir, "KMX")
    gerbil_dir = os.path.join(kmx_dir, "include", "gerbil-DataFrame")
    gerbil_build_dir = os.path.join(gerbil_dir, "build")

    if os.path.exists(kmx_dir):
        print_colored("[INFO] KMX already exists. Skipping clone.", "YELLOW")
    else:
        print_colored("[INFO] Cloning KMX repository...", "YELLOW")
        run_command(["git", "clone", "https://github.com/M-Serajian/KMX.git", kmx_dir])

    # Clone gerbil-DataFrame
    kmx_include = os.path.join(kmx_dir, "include")
    gerbil_path = os.path.join(kmx_include, "gerbil-DataFrame")
    if os.path.exists(gerbil_path):
        print_colored("[INFO] Removing existing gerbil-DataFrame (incomplete or outdated)...", "YELLOW")
        rmtree(gerbil_path)
    print_colored("[INFO] Cloning fresh gerbil-DataFrame into KMX/include...", "YELLOW")
    run_command(["git", "clone", "https://github.com/M-Serajian/gerbil-DataFrame.git"], cwd=kmx_include)

    # Build gerbil-DataFrame using conda paths
    conda_prefix = os.path.expanduser(f"~/.conda/envs/{env_name}")
    boost_include = os.path.join(conda_prefix, "include")
    boost_lib = os.path.join(conda_prefix, "lib")

    os.makedirs(gerbil_build_dir, exist_ok=True)
    print_colored("[INFO] Building gerbil-DataFrame...", "YELLOW")

    cmake_cmd = [
        "cmake",
        "..",
        f"-DBOOST_ROOT={conda_prefix}",
        f"-DCMAKE_INCLUDE_PATH={boost_include}",
        f"-DCMAKE_LIBRARY_PATH={boost_lib}",
        "-DBoost_NO_SYSTEM_PATHS=ON"
    ]

    cmake_ok = run_command(["conda", "run", "-n", env_name] + cmake_cmd, cwd=gerbil_build_dir)
    make_ok = run_command(["conda", "run", "-n", env_name, "make", "-j"], cwd=gerbil_build_dir)

    if cmake_ok and make_ok:
        print_colored("[SUCCESS] gerbil-DataFrame built successfully.", "GREEN")
    else:
        print_colored("[ERROR] Failed to build gerbil-DataFrame. Check Boost headers and CMakeLists.txt.", "RED")

def delete_environment(env_name):
    print_colored(f"[INFO] Deleting environment '{env_name}'...", "YELLOW")
    run_command(["conda", "remove", "--name", env_name, "--all", "-y"])

def parse_arguments():
    parser = argparse.ArgumentParser(description="Install/delete conda env, dependencies, and compile KMX")
    subparsers = parser.add_subparsers(dest="command", required=True)

    install_parser = subparsers.add_parser("install", help="Install environment and build KMX")
    install_parser.add_argument("--env-name", type=str, default=generate_env_name(), help="Name of the conda environment")

    delete_parser = subparsers.add_parser("delete", help="Delete the environment")
    delete_parser.add_argument("--env-name", type=str, required=True, help="Name of the conda environment to delete")

    return parser.parse_args()

def main():
    args = parse_arguments()
    env_name = args.env_name
    os.environ["ENV_NAME"] = env_name  # used by subprocesses

    if args.command == "install":
        if not create_environment(env_name):
            print_colored("[ERROR] Failed to create conda environment.", "RED")
            sys.exit(1)

        install_dependencies(env_name)
        setup_and_compile_kmx(env_name)

        kmx_src_path = os.path.join(os.getcwd(), "include", "KMX", "src")
        print_colored(f"\n[SUCCESS] Environment '{env_name}' is ready. Activate it with:", "GREEN")
        print_colored(f"conda activate {env_name}", "GREEN")
        print_colored(f"\n[INFO] To use KMX, consider adding the following to your PYTHONPATH:", "YELLOW")
        print_colored(f"export PYTHONPATH=$PYTHONPATH:{kmx_src_path}\n", "YELLOW")

    elif args.command == "delete":
        delete_environment(env_name)
        print_colored(f"[SUCCESS] Environment '{env_name}' deleted.", "GREEN")

if __name__ == "__main__":
    main()
