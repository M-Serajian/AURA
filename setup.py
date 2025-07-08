#!/usr/bin/env python3
import os
import sys
import subprocess
import platform
import argparse
from shutil import rmtree

# --- CLI Colors ---
COLORS = {
    "RED": "\033[91m",
    "GREEN": "\033[92m",
    "YELLOW": "\033[93m",
    "BLUE": "\033[94m",
    "NC": "\033[0m",
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

def create_environment():
    print_colored("[INFO] Creating environment 'resistance-predictor-env'...", "YELLOW")
    return run_command(["conda", "create", "-n", "resistance-predictor-env", "python=3.10.8", "-y"])

def install_dependencies():
    packages = {
        "xgboost": "3.0.2",
        "scipy": "1.11.4",
        "gcc_linux-64": "12.2.0",
        "gxx_linux-64": "12.2.0",
        "boost-cpp": "1.77.0",
        "cmake": "3.26.4",
        "git": "2.40.1",
        "pandas": "",
        "pyarrow": "",
    }

    for package, version in packages.items():
        pkg_str = f"{package}={version}" if version else package
        print_colored(f"[INFO] Installing {pkg_str}...", "YELLOW")
        if not run_command(["conda", "install", "-n", "resistance-predictor-env", pkg_str, "-c", "conda-forge", "-y"]):
            return False

    print_colored("[INFO] Installing pip package: simplejson...", "YELLOW")
    return run_command(["conda", "run", "-n", "resistance-predictor-env", "pip", "install", "simplejson"])

def setup_and_compile_kmx():
    base_dir = os.getcwd()
    include_dir = os.path.join(base_dir, "include")
    kmx_dir = os.path.join(include_dir, "KMX")
    gerbil_dir = os.path.join(kmx_dir, "include", "gerbil-DataFrame")
    gerbil_build_dir = os.path.join(gerbil_dir, "build")

    # Clone KMX
    if os.path.exists(kmx_dir):
        print_colored("[INFO] KMX already exists. Skipping clone.", "YELLOW")
    else:
        print_colored("[INFO] Cloning KMX repository...", "YELLOW")
        if not run_command(["git", "clone", "https://github.com/M-Serajian/KMX.git", kmx_dir]):
            return False

    # Clone gerbil-DataFrame
    kmx_include = os.path.join(kmx_dir, "include")
    gerbil_path = os.path.join(kmx_include, "gerbil-DataFrame")
    if os.path.exists(gerbil_path):
        print_colored("[INFO] Removing existing gerbil-DataFrame (incomplete or outdated)...", "YELLOW")
        rmtree(gerbil_path)
    print_colored("[INFO] Cloning fresh gerbil-DataFrame into KMX/include...", "YELLOW")
    if not run_command(["git", "clone", "https://github.com/M-Serajian/gerbil-DataFrame.git"], cwd=kmx_include):
        return False

    # Build gerbil-DataFrame
    conda_prefix = os.path.expanduser("~/.conda/envs/resistance-predictor-env")
    boost_include = os.path.join(conda_prefix, "include")
    boost_lib = os.path.join(conda_prefix, "lib")

    os.makedirs(gerbil_build_dir, exist_ok=True)
    print_colored("[INFO] Building gerbil-DataFrame...", "YELLOW")

    cmake_cmd = [
        "cmake", "..",
        f"-DBOOST_ROOT={conda_prefix}",
        f"-DCMAKE_INCLUDE_PATH={boost_include}",
        f"-DCMAKE_LIBRARY_PATH={boost_lib}",
        "-DBoost_NO_SYSTEM_PATHS=ON"
    ]

    cmake_ok = run_command(["conda", "run", "-n", "resistance-predictor-env"] + cmake_cmd, cwd=gerbil_build_dir)
    make_ok = run_command(["conda", "run", "-n", "resistance-predictor-env", "make", "-j"], cwd=gerbil_build_dir)

    return cmake_ok and make_ok

def delete_environment():
    print_colored("[INFO] Deleting environment 'resistance-predictor-env'...", "YELLOW")
    run_command(["conda", "remove", "--name", "resistance-predictor-env", "--all", "-y"])

def main():
    os_type = platform.system()
    os_version = platform.version() if os_type == "Linux" else platform.mac_ver()[0]

    if os_type == "Linux":
        print_colored(f"[INFO] Detected Linux - Kernel version: {os_version}", "BLUE")
    elif os_type == "Darwin":
        print_colored(f"[INFO] Detected macOS - Version: {os_version}", "YELLOW")
        print_colored("[WARNING] macOS is not supported yet for MTB-SHIELD setup. Exiting...", "RED")
        sys.exit(0)
    else:
        print_colored(f"[WARNING] Unsupported OS detected: {os_type}. Exiting...", "RED")
        sys.exit(0)

    parser = argparse.ArgumentParser(
        description="""
MTB-SHIELD Environment Setup Tool
----------------------------------

This setup installs all dependencies to run MTB-SHIELD, a GPU-based tool to predict 
antibiotic resistance in Mycobacterium tuberculosis for 13 antibiotics:

Isoniazid, Rifampicin, Ethambutol, Rifabutin, Ethionamide, Levofloxacin,
Moxifloxacin, Kanamycin, Amikacin, Clofazimine, Delamanid, Linezolid, Bedaquiline.

Usage:
------
Install:
    python setup.py install

Activate:
    conda activate resistance-predictor-env

Run MTB-SHIELD:
    python MTB-SHIELD.py -i <fasta_file> -o <output_csv> -t <temp_dir>
    (If -t is not provided, './temp' will be used by default)

Deactivate:
    conda deactivate

Delete environment:
    python setup.py delete
""",
        formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument("command", choices=["install", "delete"], help="Install or delete the conda environment")
    args = parser.parse_args()

    for var in ("PYTHONPATH", "PYTHONHOME"):
        os.environ.pop(var, None)

    if args.command == "install":
        if not create_environment():
            print_colored("[ERROR] Environment creation failed.", "RED")
            sys.exit(1)

        if not install_dependencies():
            print_colored("[ERROR] Failed to install all dependencies.", "RED")
            delete_environment()
            sys.exit(1)

        if not setup_and_compile_kmx():
            print_colored("[ERROR] Failed to build gerbil-DataFrame.", "RED")
            rmtree(os.path.join("include", "KMX"), ignore_errors=True)
            delete_environment()
            print_colored("[SUGGESTION] Please retry: python setup.py delete", "RED")
            sys.exit(1)

        print_colored("\n[SUCCESS] Environment 'resistance-predictor-env' is created successfully!", "BLUE")
        print_colored("conda activate resistance-predictor-env", "BLUE")
        print_colored("\nUsage:", "BLUE")
        print_colored("python MTB-SHIELD.py -i <path_to_fasta> -o <path_to_output_csv> -t <temp_dir>", "BLUE")
        print_colored("If -t is not provided, './temp' will be used by default.", "BLUE")
        print_colored("\nTo deactivate the environment:", "BLUE")
        print_colored("conda deactivate", "BLUE")
        print_colored("\nTo delete the environment:", "BLUE")
        print_colored("python setup.py delete", "BLUE")

    elif args.command == "delete":
        delete_environment()
        print_colored("[SUCCESS] Environment 'resistance-predictor-env' deleted.", "GREEN")

if __name__ == "__main__":
    main()
