#!/usr/bin/env python3


import argparse
import os
import sys

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


def validate_paths(input_fasta: str, output_directory: str):
    """
    Validate existence of input FASTA file and create output directory if needed.

    Args:
        input_fasta (str): Path to the input FASTA file.
        output_directory (str): Directory for storing the output CSV.
    """
    if not os.path.isfile(input_fasta):
        print(f"Error: Input FASTA file not found: {input_fasta}")
        sys.exit(1)
    
    if not os.path.exists(output_directory):
        os.makedirs(output_directory)


def main():
    """
    Main execution workflow:
    - Parses arguments
    - Validates inputs
    - Loads appropriate processing libraries
    - Handles FASTA processing (to be implemented)
    """
    args = parse_command_line_arguments()

    if args.gpu:
        print("Running in GPU mode.")
    else:
        print("Running in CPU mode (default).")

    validate_paths(args.fasta, args.output)
    load_processing_libraries(args.gpu)

    # === Placeholder for FASTA processing logic ===
    print("FASTA processing logic starts here.")

    # Example output
    output_path = os.path.join(args.output, "output.csv")
    result_df = pd.DataFrame({"Status": ["Processed Placeholder"]})
    result_df.to_csv(output_path, index=False)

    print(f"Output written to: {output_path}")
    print("Execution completed.")


if __name__ == "__main__":
    main()
