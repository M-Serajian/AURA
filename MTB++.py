#!/usr/bin/env python3

import argparse
import os
import sys
import time  # <---- ADD THIS FOR PROFILING

# FIRST: fix sys.path BEFORE any project imports
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT_DIR)

# Delay the heavy import
# from include.GPU_CSR_Kmer.src.run_gerbil import single_genome_kmer_extractor

# Global data processing libraries — assigned dynamically
np = None
pd = None

def parse_command_line_arguments():
    parser = argparse.ArgumentParser(
        description="Convert a FASTA file into CSV using CPU or GPU."
    )
    parser.add_argument("-f", "--fasta", required=True, type=str, help="Path to the input FASTA file.")
    parser.add_argument("-o", "--output", required=True, type=str, help="Directory for output CSV file.")
    parser.add_argument("-g", "--gpu", action="store_true", help="Enable GPU processing using RAPIDS cuDF/CuPy.")
    return parser.parse_args()

def load_processing_libraries(use_gpu: bool):
    global np, pd

    if use_gpu:
        import cupy as np
        import cudf as pd
        print("GPU mode enabled. Using CuPy and cuDF for processing.")
    else:
        import numpy as np
        import pandas as pd
        print("CPU mode enabled (default). Using NumPy and Pandas for processing.")

def main():
    total_start = time.time()

    # Step 1: Parse arguments
    start = time.time()
    args = parse_command_line_arguments()
    print(f"[Profile] Argument parsing took {time.time() - start:.4f} seconds.")

    # Step 2: Load libraries
    start = time.time()
    import cupy 
    import cudf
    print(f"[Profile] Library loading took {time.time() - start:.4f} seconds.")

    # Step 3: Import heavy modules
    start = time.time()
    from include.GPU_CSR_Kmer.src.run_gerbil import single_genome_kmer_extractor
    print(f"[Profile] Heavy module import took {time.time() - start:.4f} seconds.")

    # (Optional) Print total
    print(f"[Profile] Total execution until here took {time.time() - total_start:.4f} seconds.")

if __name__ == "__main__":
    main()
