#!/usr/bin/env python3
import cudf
import cupy as cp
import pandas as pd
from src.dataframe_merge.dataframe_merge import dataframe_kmer_refrence_merge
from src.dataframe_merge.dataframe_merge import fill_NA_zero
from src.dataframe_npy_trasformation.dataframe_npy_trasformation import save_df_as_npy
import os
import copy
import gc  # Garbage collector interface
import cupyx
from src.config.config import Config
import time
config = Config.get_instance()














# def dataframe_merge_CSR(number_of_samples,base,input_list,source_dataframe_dir):
#     print(f"Testing dataframe merge function of CUDF Nvidia on {number_of_samples} genomes of CRyPTIC MTB data:",flush=True)
#     # Example data for df_union ( refrence of all the unique kmers accross all the Gerbil outputs)
#     source_dataframe = cudf.read_csv(source_dataframe_dir)
#     source_dataframe = source_dataframe[["K-mer"]]
#     source_dataframe = source_dataframe.sort_values('K-mer')

#     #a unique K-mer index for mapping each new kmer
#     kmer_to_index = cudf.Series(source_dataframe.index, index=source_dataframe['K-mer'])
    
#     # # Testing => done for approx. 50 Millon kmers => Successful!
#     # # Check for duplicates in the index of kmer_to_index
#     # if kmer_to_index.index.duplicated().any():
#     #     print("There are duplicate k-mers in the index.")
#     # else:
#     #     print("All k-mers are unique in the index.")

#     # Test done! 
#     # print("Shape of source dataframe:",flush=True)
#     # print(source_dataframe.shape,flush=True)
#     # print(source_dataframe.head(10))
#     # print("-------------------------------------------",flush=True)
#     # Mapping K-mers to unique Kmers
#     source_dataframe['K-mer'] = source_dataframe['K-mer'].map(kmer_to_index)

#     # test done!
#     # print("Shape of source dataframe:",flush=True)
#     # print(source_dataframe.shape,flush=True)
#     # print("-------------------------------------------",flush=True)
#     # source_dataframe = source_dataframe[["K-mer"]]
#     # print(source_dataframe.shape)

#     #Creating the CSR
#     #union_dataframe= (copy.copy(source_dataframe)).T

#     # Creating a new DataFrame where each value in 'K-mer' becomes a separate column
#     k_mer_array = source_dataframe['K-mer'].values

#     # Create a new DataFrame with this array as a row
#     transposed_df = cudf.DataFrame([k_mer_array])

#     # Optionally, rename columns to reflect original indices or any specific naming scheme
#     transposed_df.columns = [f'Col_{i}' for i in range(len(transposed_df.columns))]



def dataframe_merge_CSR(number_of_samples, base, input_list):

    def update_csr_matrix(row, kmer_index, frequency, sample_number):
        if cp.isfinite(frequency):
            # Directly update the CSR matrix
            # Note: Accessing and updating a csr_matrix in such a function is not directly possible as
            # shared memory objects like this aren't supported in `apply_rows`.
            # We typically need to return values and handle updates outside this function.
            # This is a limitation we're currently working around conceptually.
            csr_matrix[sample_number, kmer_index] = frequency


    print(f"Testing dataframe merge function of CUDF Nvidia on {number_of_samples} genomes of CRyPTIC MTB data:", flush=True)
    # source_dataframe = source_dataframe.sort_values('K-mer')

    #source_dataframe = config.source_dataframe

    kmer_to_index = cudf.Series(config.source_dataframe.index, index=config.source_dataframe['K-mer'])
    #source_dataframe['K-mer'] = source_dataframe['K-mer'].map(kmer_to_index)
    max_index = kmer_to_index.max()


    num_rows = number_of_samples  # This should match the number of vertical dataframes you expect
    num_cols = max_index+1   # This should match the number of features in each dataframe

    # Create an empty CSR matrix
    csr_matrix = cupyx.scipy.sparse.csr_matrix((num_rows, num_cols), dtype=cp.float32)



    with open(input_list, 'r') as file:
            directories = []
            for i, line in enumerate(file):
                if i >= number_of_samples:  # Stop reading after the first 100 lines
                    break
                line = line.strip()
                if line:  # Only add non-empty lines
                    directories.append(line)


    for sample_number, directory in enumerate(directories, start=0):
        print(sample_number,flush=True)
        print(directory,flush=True)
        # Extract the parent directory containing "ERR"
        parent_directory = os.path.basename(os.path.dirname(directory))
        mid_df_dir = base + parent_directory + ".csv"
        mid_dataframe = cudf.read_csv(mid_df_dir)
        #print(mid_dataframe.head(29),flush=True)
        
        merged_dataframe = (cudf.merge(config.source_dataframe, mid_dataframe, on='K-mer', how='left'))
        merged_dataframe = merged_dataframe.dropna(subset=['Frequency'])
        print(merged_dataframe.shape[0],flush=True)
        # print(merged_dataframe['Frequency'].count())
        merged_dataframe['K-mer'] = merged_dataframe['K-mer'].map(kmer_to_index)
        
        #merged_dataframe = merged_dataframe.dropna(subset=['Frequency'])
        # print(merged_dataframe.head(100),flush=True)
        # print(merged_dataframe['Frequency'].count())
        # break

        frequencies = cp.asarray(merged_dataframe['Frequency'].values)

        kmer_indices = cp.asarray(cp.asarray(merged_dataframe['K-mer'].values))
        del merged_dataframe

        # Update the CSR matrix using CuPy (handling non-NaNs and ensuring they are finite)
        t1=time.time()
        for kmer_index, frequency in zip(kmer_indices, frequencies):
            print("ok",flush=True)
            csr_matrix[sample_number, kmer_index] = frequency

        t2=time.time()
        print(t2-t1)
        break