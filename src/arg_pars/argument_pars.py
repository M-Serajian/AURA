#!/usr/bin/env python3
import argparse

def parse_arguments():
    parser = argparse.ArgumentParser(description="GPU Acceleration")
    parser.add_argument("--debug", action="store_true", help="Activate debug mode")
    parser.add_argument("--test", action="store_true", help="Activate test mode")
    parser.add_argument("-i", "--input", type=str, required=True, help="Input dataframe (csv format)")
    args = parser.parse_args()
    return args