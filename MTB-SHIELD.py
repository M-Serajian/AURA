#!/usr/bin/env python3

"""
AMR Predictor Script
=====================

This script predicts antimicrobial resistance (AMR) from a given genome FASTA file.
It supports GPU acceleration using RAPIDS (cuDF and CuPy) or CPU fallback (Pandas and NumPy).
The prediction is based on pre-trained machine learning models (XGBoost, Logistic Regression,
Random Forest, or Linear SVM) using k-mer frequency vectors.

Main Tasks:
-----------
1. Extract k-mers from the genome using a CUDA-accelerated Gerbil wrapper.
2. Map k-mers to the top features used during training.
3. Predict resistance using a serialized model loaded from JSON.
4. Output the results as a CSV file.

Author: [MSRJN]
Date: [May 2025]
"""

import argparse
import os
import sys
import random
import string
import xgboost as xgb
import scipy.sparse as sp
import json
import base64

# Ensure imports from project directory
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
CLASSIFIER_ROOT_DIR = os.path.join(ROOT_DIR, "data", "trained_models")

from include.KMX.src.run_gerbil import single_genome_kmer_extractor

# Global data processing libraries (np and pd), set dynamically based on --gpu flag
np = None
pd = None



# --- CLI color codes ---
GREEN = "\033[92m"
NC = "\033[0m"  # No Color


def parse_command_line_arguments():
    """
    Parse and validate command-line arguments.
    Returns:
        argparse.Namespace: Parsed arguments
    """
    parser = argparse.ArgumentParser(
        description="Predict antimicrobial resistance from a FASTA file using trained models."
    )
    parser.add_argument("-i", "--input", required=True, type=str,
                        help="Path to the input FASTA file.")
    parser.add_argument("-o", "--output", required=True, type=str,
                        help="Path to the output CSV file (must end with .csv).")
    parser.add_argument("-g", "--gpu", action="store_true",
                        help="Enable GPU processing using RAPIDS cuDF/CuPy.")
    parser.add_argument("-t", "--temp-directory", type=str,
                        default=os.path.join(ROOT_DIR, "temp"),
                        help=f"Temporary directory (default: {os.path.join(ROOT_DIR, 'temp')}). "
                             "Must have at least 5 GB free space.")

    args = parser.parse_args()

    if not args.output.lower().endswith(".csv"):
        parser.error("The output file must end with `.csv`.")

    return args


def load_processing_libraries(use_gpu: bool):
    """
    Dynamically import data processing libraries based on GPU flag.
    Sets global np and pd accordingly.
    """
    global np, pd

    if use_gpu:
        import cupy as np
        import cudf as pd
        print("GPU mode enabled. Using CuPy and cuDF for processing.")
    else:
        import numpy as np
        import pandas as pd
        print("CPU mode enabled (default). Using NumPy and Pandas.")


# Parse command-line arguments and load processing libraries accordingly
args = parse_command_line_arguments()
load_processing_libraries(args.gpu)


class ResistancePredictor:
    """
    Handles model deserialization and prediction logic.
    Supports multiple classifier types and input formats (dense or CSR).
    """
    def __init__(self, model_data: dict):
        self.model_data = model_data

        self.classifier_type = model_data["classifier_type"]
        self.threshold = float(model_data["threshold"])
        self.kmer_length = int(model_data["kmer_length"])
        self.number_of_features = int(model_data["number_of_features"])
        self.antibiotic = model_data["antibiotic"]
        self.csr_input = bool(model_data["CSR"])

        self.model = self._load_model()

    def _load_model(self):
        """
        Deserialize model based on classifier type.
        Returns:
            Deserialized model object or parameter dictionary.
        """
        if self.classifier_type == "XGB":
            booster = xgb.Booster()
            raw_bytes = base64.b64decode(self.model_data["model_detail"])
            booster.load_model(bytearray(raw_bytes))
            return booster

        elif self.classifier_type == "RF":
            model = RandomForestClassifier()
            raw_bytes = base64.b64decode(self.model_data["model_detail"])
            model.deserialize(raw_bytes)
            return model

        elif self.classifier_type in ("LR", "linearSVM"):
            return {
                "coef": np.array(self.model_data["coef"]),
                "intercept": np.array(self.model_data["intercept"]),
                "classes": np.array(self.model_data["classes"])
            }

        raise ValueError(f"Unsupported classifier type: {self.classifier_type}")

    def _dense_row_to_cpu_csr(self, dense_row: np.ndarray) -> sp.csr_matrix:
        """
        Convert dense row to SciPy-compatible CSR format (for XGBoost/RandomForest).
        """
        if dense_row.ndim != 2 or dense_row.shape[0] != 1:
            raise ValueError(f"Expected shape (1, N), got {dense_row.shape}")

        dense_row = dense_row.flatten()
        nonzero_indices = np.nonzero(dense_row)[0]
        nonzero_values = dense_row[nonzero_indices]

        values_cpu = nonzero_values.get() if hasattr(nonzero_values, "get") else nonzero_values
        indices_cpu = nonzero_indices.get() if hasattr(nonzero_indices, "get") else nonzero_indices
        row_ptr = np.array([0, len(values_cpu)], dtype=np.int32)

        return sp.csr_matrix((values_cpu, indices_cpu, row_ptr), shape=(1, dense_row.size))

    def predict(self, input_row: np.ndarray) -> str:
        """
        Predict binary class label: "Resistant" or "Susceptible".
        """
        if self.csr_input:
            input_row = self._dense_row_to_cpu_csr(input_row)

        if self.classifier_type == "XGB":
            dmatrix = xgb.DMatrix(input_row)
            prob = float(self.model.predict(dmatrix)[0])
            return "Resistant" if prob >= self.threshold else "Susceptible"

        elif self.classifier_type == "RF":
            prob = float(self.model.predict_proba(input_row)[0, 1])
            return "Resistant" if prob >= self.threshold else "Susceptible"

        elif self.classifier_type == "LR":
            prob = 1 / (1 + np.exp(-(np.dot(input_row, self.model["coef"].T) + self.model["intercept"])))
            return "Resistant" if prob[0, 0] >= self.threshold else "Susceptible"

        elif self.classifier_type == "linearSVM":
            margin = np.dot(input_row, self.model["coef"].T) + self.model["intercept"]
            return "Resistant" if margin[0, 0] > 0 else "Susceptible"

        raise ValueError(f"Unsupported classifier type: {self.classifier_type}")

    def predict_proba(self, input_row: np.ndarray) -> float:
        """
        Return probability of resistance (if supported).
        """
        if self.classifier_type == "linearSVM":
            raise NotImplementedError("Probability is not supported for LinearSVM.")

        if self.csr_input:
            input_row = self._dense_row_to_cpu_csr(input_row)

        if self.classifier_type == "XGB":
            return float(self.model.predict(xgb.DMatrix(input_row))[0])
        elif self.classifier_type == "RF":
            return float(self.model.predict_proba(input_row)[0, 1])
        elif self.classifier_type == "LR":
            prob = 1 / (1 + np.exp(-(np.dot(input_row, self.model["coef"].T) + self.model["intercept"])))
            return float(prob[0, 0])
        else:
            raise ValueError(f"Unsupported classifier type: {self.classifier_type}")


def main():
    """
    Main pipeline: loads models, extracts k-mers, predicts resistance, and writes results to CSV.
    """
    output_csv_header = ["Antibiotic"]
    predictions = ["AMR Classification"]

    # Temporary directory with unique name
    temporary_files_prefix = ''.join(random.choices(string.ascii_letters, k=30))
    temporary_files_dir = os.path.join(args.temp_directory, temporary_files_prefix)
    if not temporary_files_dir.endswith(os.sep):
        temporary_files_dir += os.sep
    os.makedirs(temporary_files_dir, exist_ok=True)

    # Identify antibiotics with both model (.json) and feature list (.parquet)
    antibiotics = sorted({f.split('.')[0] for f in os.listdir(CLASSIFIER_ROOT_DIR) if f.endswith('.parquet')} &
                         {f.split('.')[0] for f in os.listdir(CLASSIFIER_ROOT_DIR) if f.endswith('.json')})

    classifiers = {}
    for antibiotic in antibiotics:
        model_path = os.path.join(CLASSIFIER_ROOT_DIR, f"{antibiotic}.json")
        with open(model_path, "r") as f:
            model_data = json.load(f)
            globals()[antibiotic] = model_data

            k = int(model_data["kmer_length"])
            if k not in classifiers:
                classifiers[k] = []
            classifiers[k].append({"Antibiotic": antibiotic})

    for kmer_length, antibiotic_list in classifiers.items():
        output_path = os.path.join(temporary_files_dir, "gerbil_output.csv")

        # Extract k-mers from FASTA using GPU or CPU
        single_genome_kmer_extractor(kmer_length, temporary_files_dir, output_path, args.input, False, args.gpu)
        genome_kmer_dataframe = pd.read_csv(output_path)

        for entry in antibiotic_list:
            antibiotic_abbr = entry["Antibiotic"]
            model_data = globals()[antibiotic_abbr]
            output_csv_header.append(antibiotic_abbr)

            # Load top k-mers used in model
            source_kmer_list = pd.read_parquet(os.path.join(CLASSIFIER_ROOT_DIR, f"{antibiotic_abbr}.parquet"))
            source_kmer_list = source_kmer_list.head(model_data["number_of_features"])

            # Preserve original row order during merge
            source_kmer_list['_merge_order'] = np.arange(len(source_kmer_list))
            source_kmer_list = pd.merge(source_kmer_list, genome_kmer_dataframe, on='K-mer', how='left', sort=False)
            source_kmer_list = source_kmer_list.sort_values('_merge_order').drop(columns=['_merge_order']).reset_index(drop=True)

            # Fill missing k-mers with frequency 0
            source_kmer_list['Frequency'] = source_kmer_list['Frequency'].fillna(0).astype(int)

            # Prepare feature vector
            dense_array = np.asarray(source_kmer_list['Frequency'].values, dtype=np.float32).reshape(1, -1)

            # Predict using pre-trained model
            predictor = ResistancePredictor(model_data)
            prediction = predictor.predict(dense_array)
            predictions.append(prediction)

        del genome_kmer_dataframe  # Free memory

    # Clean up temp directory
    os.system(f'rm -rf {temporary_files_dir}')

    # Save final result as CSV
    result_df = pd.DataFrame([predictions], columns=output_csv_header)
    result_df.to_csv(args.output, index=False)
    print(f"\n{GREEN}Resistance profiling for {args.input} is available at: {args.output}{NC}")


if __name__ == "__main__":
    main()