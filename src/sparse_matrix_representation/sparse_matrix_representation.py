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
import sys
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



# def dataframe_merge_CSR(number_of_samples, base, input_list):


#     print(f"Testing dataframe merge function of CUDF Nvidia on {number_of_samples} genomes of CRyPTIC MTB data:", flush=True)
#     # source_dataframe = source_dataframe.sort_values('K-mer')

#     #source_dataframe = config.source_dataframe

#     kmer_to_index = cudf.Series(config.source_dataframe.index, index=config.source_dataframe['K-mer'])
#     #source_dataframe['K-mer'] = source_dataframe['K-mer'].map(kmer_to_index)
#     max_index = kmer_to_index.max()


#     num_rows = number_of_samples  # This should match the number of vertical dataframes you expect
#     num_cols = max_index+1   # This should match the number of features in each dataframe

#     # Create an empty CSR matrix
#     csr_matrix = cupyx.scipy.sparse.csr_matrix((num_rows, num_cols), dtype=cp.float32)



#     with open(input_list, 'r') as file:
#             directories = []
#             for i, line in enumerate(file):
#                 if i >= num_rows:  # Stop reading after the first 100 lines
#                     break
#                 line = line.strip()
#                 if line:  # Only add non-empty lines
#                     directories.append(line)



def dataframe_merge_CSR_parquet(parquet_directories_list_path, source_dataframe_dir):

    #time profiling variables:
    reading_parquet_files_time=0
    merging_processing_time=0

    #list of the parquet files gene
    with open(parquet_directories_list_path, 'r') as file:
    # Read all lines into a list and strip newline characters
        parquet_directories = [line.strip() for line in file.readlines() if line.strip()]

    # Dataframe of the all unique kmers filterd accross all the genomes
    t0=time.time() #start_time
    source_dataframe = cudf.read_parquet(source_dataframe_dir)
    reading_parquet_files_time=reading_parquet_files_time+ time.time()- t0

    source_dataframe = source_dataframe[["K-mer"]]

    # CST matrix pointers
    row_pnt = [0]
    column_idx =cp.empty(int(0.003*len(source_dataframe)*len(parquet_directories)), dtype=cp.uint32) # 
    vals = cp.zeros_like(column_idx,dtype=cp.float32)
    current_position = 0

    for i, directory in enumerate(parquet_directories):
        df_comp = source_dataframe.reset_index()
        t0=time.time()
        df_parquet = cudf.read_parquet(directory)
        reading_parquet_files_time=reading_parquet_files_time+ (time.time()- t0)


        #start time
        t0=time.time()
        df_comp = df_comp.merge(df_parquet , on = 'K-mer', how='right') # left is slower (keeps all the k-mers in the source dataframe)
        df_comp.dropna(inplace=True)
        idx = df_comp['index'].values.astype(cp.uint32)
        val = df_comp['Frequency'].values
        # if not idx.any():
        #     print('no idx available')
        size = idx.size
        column_idx[current_position:current_position + size] = idx
        vals[current_position:current_position + size] = val
        current_position+=size
        row_pnt.append(current_position)
        merging_processing_time=merging_processing_time+ time.time()-t0

        if (i%10==0):
            print(i,flush=True)
            # current_time=time.time()
            # print(f"Total time for {i+1} genomes in S: {round((current_time-t0),2)}")
            # total_memory_usage_gb = (sum(sys.getsizeof(item) for item in row_pnt) + sum(sys.getsizeof(item) for item in column_idx) + sum(sys.getsizeof(item) for item in vals) + sys.getsizeof(row_pnt) + sys.getsizeof(column_idx) + sys.getsizeof(vals)) / (1024**3)
            # print("Total Memory Usage in GB:", total_memory_usage_gb, flush=True)
    last_value = row_pnt[-1]
    print(f"Merging time: {merging_processing_time}",flush=True)
    print(f"Reading time: {reading_parquet_files_time}",flush=True)
    return cp.asarray(row_pnt),column_idx[:last_value],vals[:last_value]






def dataframe_merge_CSR_csv(csv_directories_list_path, source_dataframe_dir):

    #time profiling variables:
    # reading_csv_files_time=0
    # merging_processing_time=0

    #list of the csv files gene
    with open(csv_directories_list_path, 'r') as file:
    # Read all lines into a list and strip newline characters
        csv_directories = [line.strip() for line in file.readlines() if line.strip()]

    # Dataframe of the all unique kmers filterd accross all the genomes
    # t0=time.time() #start_time
    source_dataframe = cudf.read_csv(source_dataframe_dir)
    # reading_csv_files_time=reading_csv_files_time+ time.time()- t0

    source_dataframe = source_dataframe[["K-mer"]]

    # CST matrix pointers
    row_pnt = [0]
    column_idx =cp.empty(int(0.0025*len(source_dataframe)*len(csv_directories)), dtype=cp.uint32) # 
    vals = cp.zeros_like(column_idx,dtype=cp.float32)
    current_position = 0

    for i, directory in enumerate(csv_directories):
        df_comp = source_dataframe.reset_index()
        # t0=time.time()
        df_csv = cudf.read_csv(directory)
        # reading_csv_files_time=reading_csv_files_time+ (time.time()- t0)


        #start time
        # t0=time.time()
        df_comp = df_comp.merge(df_csv , on = 'K-mer', how='right') # left is slower (keeps all the k-mers in the source dataframe)
        df_comp.dropna(inplace=True)
        idx = df_comp['index'].values.astype(cp.uint32)
        val = df_comp['Frequency'].values
        # if not idx.any():
        #     print('no idx available')
        size = idx.size
        column_idx[current_position:current_position + size] = idx
        vals[current_position:current_position + size] = val
        current_position+=size
        row_pnt.append(current_position)
        # merging_processing_time=merging_processing_time+ time.time()-t0

        if (i%1000==0):
            print(i,flush=True)
            # current_time=time.time()
            # print(f"Total time for {i+1} genomes in S: {round((current_time-t0),2)}")
            # total_memory_usage_gb = (sum(sys.getsizeof(item) for item in row_pnt) + sum(sys.getsizeof(item) for item in column_idx) + sum(sys.getsizeof(item) for item in vals) + sys.getsizeof(row_pnt) + sys.getsizeof(column_idx) + sys.getsizeof(vals)) / (1024**3)
            # print("Total Memory Usage in GB:", total_memory_usage_gb, flush=True)
    last_value = row_pnt[-1]
    # print(f"Merging time: {merging_processing_time}",flush=True)
    # print(f"Reading time: {reading_csv_files_time}",flush=True)
    return cp.asarray(row_pnt),column_idx[:last_value],vals[:last_value]





def dataframe_merge_CSR_csv_input_list(csv_directories, source_dataframe_dir):


    source_dataframe = cudf.read_csv(source_dataframe_dir)

    source_dataframe = source_dataframe[["K-mer"]]

    # CST matrix pointers
    row_pnt = [0]
    column_idx =cp.empty(int(0.8*len(source_dataframe)*len(csv_directories)), dtype=cp.uint32) # 
    vals = cp.zeros_like(column_idx,dtype=cp.float32)
    current_position = 0

    for i, directory in enumerate(csv_directories):
        df_comp = source_dataframe.reset_index()
        
        df_csv = cudf.read_csv(directory)
        
        print("Number of rows src dataframe:", df_comp.shape[0])

        print("Number of rows example:", df_csv.shape[0])

        df_comp = df_comp.merge(df_csv , on = 'K-mer', how='right') # left is slower (keeps all the k-mers in the source dataframe)
        df_comp.dropna(inplace=True)
        print("Number of rows:", df_comp.shape[0])
        print("----------")
        idx = df_comp['index'].values.astype(cp.uint32)
        val = df_comp['Frequency'].values
        # if not idx.any():
        #     print('no idx available')
        size = idx.size
        column_idx[current_position:current_position + size] = idx
        vals[current_position:current_position + size] = val
        current_position+=size
        row_pnt.append(current_position)

        if (i%10==0):
            print(i,flush=True)


    last_value = row_pnt[-1]
    df_comp = source_dataframe.reset_index()

    return cp.asarray(row_pnt),column_idx[:last_value],vals[:last_value]





def initial_dataframe_merge_CSR_csv(csv_directories, source_dataframe_dir):

    source_dataframe = cudf.read_csv(source_dataframe_dir)
    # reading_csv_files_time=reading_csv_files_time+ time.time()- t0

    source_dataframe = source_dataframe[["K-mer"]]

    # CST matrix pointers
    row_pnt = [0]
    column_idx =cp.empty(int(0.004*len(source_dataframe)*len(csv_directories)), dtype=cp.uint32) # 
    vals = cp.zeros_like(column_idx,dtype=cp.float32)
    current_position = 0

    for i, directory in enumerate(csv_directories):
        df_comp = source_dataframe.reset_index()
        # t0=time.time()
        df_csv = cudf.read_csv(directory)
        # reading_csv_files_time=reading_csv_files_time+ (time.time()- t0)

        #start time
        # t0=time.time()
        df_comp = df_comp.merge(df_csv , on = 'K-mer', how='right') # left is slower (keeps all the k-mers in the source dataframe)

        df_comp.dropna(inplace=True)

        idx = df_comp['index'].values.astype(cp.uint32)
        val = df_comp['Frequency'].values
        # if not idx.any():
        #     print('no idx available')
        size = idx.size
        column_idx[current_position:current_position + size] = idx
        vals[current_position:current_position + size] = val
        current_position+=size
        row_pnt.append(current_position)
        # merging_processing_time=merging_processing_time+ time.time()-t0

        if (i%1000==0):
            print(i,flush=True)
            # current_time=time.time()
            # print(f"Total time for {i+1} genomes in S: {round((current_time-t0),2)}")
            # total_memory_usage_gb = (sum(sys.getsizeof(item) for item in row_pnt) + sum(sys.getsizeof(item) for item in column_idx) + sum(sys.getsizeof(item) for item in vals) + sys.getsizeof(row_pnt) + sys.getsizeof(column_idx) + sys.getsizeof(vals)) / (1024**3)
            # print("Total Memory Usage in GB:", total_memory_usage_gb, flush=True)
    last_value = row_pnt[-1]



    matrix = cupyx.scipy.sparse.csr_matrix((vals[:last_value],column_idx[:last_value],cp.asarray(row_pnt)))
    vals_memory = matrix.data.nbytes
    column_idx_memory = matrix.indices.nbytes
    row_pnt_memory = matrix.indptr.nbytes

    total_memory = vals_memory + column_idx_memory + row_pnt_memory
    print(f"Total used memory for csr matrix is: {total_memory}")
    # print(f"Merging time: {merging_processing_time}",flush=True)
    # print(f"Reading time: {reading_csv_files_time}",flush=True)
    return matrix