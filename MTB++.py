#!/usr/bin/env python3


import argparse
import os
import sys

# FIRST: fix sys.path BEFORE any project imports
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT_DIR)


from include.GPU_CSR_Kmer.src.run_gerbil import single_genome_kmer_extractor



# Global data processing libraries — assigned dynamically
np = None
pd = None


def parse_command_line_arguments():
    """
    Parse and validate command-line arguments using argparse.
    
    Returns:
        argparse.Namespace: Parsed arguments object.
    """
    parser = argparse.ArgumentParser(
        description="Convert a FASTA file into CSV using CPU or GPU."
    )
    parser.add_argument(
        "-f", "--fasta", required=True, type=str, help="Path to the input FASTA file."
    )
    parser.add_argument(
        "-o", "--output", required=True, type=str, help="Directory for output CSV file."
    )
    parser.add_argument(
        "-g", "--gpu", action="store_true", help="Enable GPU processing using RAPIDS cuDF/CuPy."
    )
    return parser.parse_args()


def load_processing_libraries(use_gpu: bool):
    """
    Dynamically import libraries depending on CPU or GPU mode.

    Args:
        use_gpu (bool): If True, loads CuPy and cuDF for GPU processing.
    """
    global np, pd
    try:
        if use_gpu:
            import cupy as np
            import cudf as pd
            print("GPU mode enabled. Using CuPy and cuDF for processing.")
        else:
            import numpy as np
            import pandas as pd
            print("CPU mode enabled (default). Using NumPy and Pandas for processing.")
    except ImportError as e:
        print(f"Library import failed: {e}")
        sys.exit(1)



def main():

    args = parse_command_line_arguments()

    if args.gpu:
        print("Running in GPU mode.")
    else:
        print("Running in CPU mode (default).")

    load_processing_libraries(args.gpu)





if __name__ == "__main__":
    main()
