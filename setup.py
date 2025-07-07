#!/usr/bin/env python3
import os
import sys
import subprocess
import platform
import random
import string
import argparse

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
    # Step 1: System dependencies
    print_colored("[INFO] Installing system-level dependencies with apt...", "YELLOW")
    apt_packages = [
        "cmake", "git", "libboost-all-dev", "libz3-dev", "libbz2-dev"
    ]
    if platform.system() == "Linux":
        run_command(["sudo", "apt-get", "update"])
        run_command(["sudo", "apt-get", "install", "-y"] + apt_packages)
    else:
        print_colored("[WARNING] Skipping apt install; not a Linux system.", "YELLOW")

    # Step 2: Conda packages
    packages = {
        "xgboost": "1.6.2",
        "scipy": "1.11.4",
        "gcc": "12.2.0",
        "boost": "1.77",
        "git": "2.40.1"
    }

    for package, version in packages.items():
        print_colored(f"[INFO] Installing {package}={version}...", "YELLOW")
        run_command(["conda", "install", "-n", env_name, f"{package}={version}", "-c", "conda-forge", "-y"])

    # Step 3: pip dependencies
    print_colored("[INFO] Installing pip package: simplejson...", "YELLOW")
    if run_command(["conda", "run", "-n", env_name, "pip", "install", "simplejson"]):
        print_colored("[SUCCESS] simplejson installed successfully.", "GREEN")
    else:
        print_colored("[WARNING] Failed to install simplejson.", "RED")

    # Step 4: Git check
    print_colored("[INFO] Verifying git installation...", "YELLOW")
    run_command(["conda", "run", "-n", env_name, "git", "--version"])

def setup_and_compile_kmx():
    base_dir = os.getcwd()
    include_dir = os.path.join(base_dir, "include")
    kmx_dir = os.path.join(include_dir, "KMX")
    gerbil_dir = os.path.join(kmx_dir, "include", "gerbil-DataFrame")
    gerbil_build_dir = os.path.join(gerbil_dir, "build")

    # Clone KMX repo
    if not os.path.exists(kmx_dir):
        print_colored("[INFO] Cloning KMX repository...", "YELLOW")
        run_command(["git", "clone", "https://github.com/M-Serajian/KMX.git", kmx_dir])
    else:
        print_colored("[INFO] KMX already exists. Skipping clone.", "YELLOW")

    # Clone gerbil-DataFrame inside KMX/include
    kmx_include = os.path.join(kmx_dir, "include")
    if not os.path.exists(os.path.join(kmx_include, "gerbil-DataFrame")):
        print_colored("[INFO] Cloning gerbil-DataFrame inside KMX/include...", "YELLOW")
        run_command(["git", "clone", "https://github.com/M-Serajian/gerbil-DataFrame.git"], cwd=kmx_include)
    else:
        print_colored("[INFO] gerbil-DataFrame already exists. Skipping clone.", "YELLOW")

    # Build gerbil-DataFrame
    os.makedirs(gerbil_build_dir, exist_ok=True)
    print_colored("[INFO] Building gerbil-DataFrame...", "YELLOW")
    if run_command(["cmake", ".."], cwd=gerbil_build_dir) and run_command(["make", "-j"], cwd=gerbil_build_dir):
        print_colored("[SUCCESS] gerbil-DataFrame built successfully.", "GREEN")
    else:
        print_colored("[ERROR] Failed to build gerbil-DataFrame.", "RED")

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

    if args.command == "install":
        env_name = args.env_name
        if not create_environment(env_name):
            print_colored("[ERROR] Failed to create conda environment.", "RED")
            sys.exit(1)

        install_dependencies(env_name)
        setup_and_compile_kmx()

        print_colored(f"\n[SUCCESS] Environment '{env_name}' is ready. Activate it with:", "GREEN")
        print_colored(f"conda activate {env_name}\n", "GREEN")

    elif args.command == "delete":
        delete_environment(args.env_name)
        print_colored(f"[SUCCESS] Environment '{args.env_name}' deleted.", "GREEN")

if __name__ == "__main__":
    main()
