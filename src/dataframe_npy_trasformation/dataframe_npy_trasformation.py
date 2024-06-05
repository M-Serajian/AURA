#!/usr/bin/env python3
import numpy as np
import os
import cudf
import pandas as pd
# takes df, changes the 
def save_df_as_npy(df,address,main_name):

    # Attempt to encode the first row
    # df.iloc[0, :] = df.iloc[0, :].apply(lambda x: x.encode('utf-8') if isinstance(x, str) else x)
    # # Attempt to encode the first column (Index), applying encoding only to strings
    #transforming all the kmers to ingegers
    #df.iloc[0:, 0] = df.iloc[0:, 0].apply(lambda x: x.encode('utf-8') if isinstance(x, str) else x)
    #data_array = df.to_pandas().to_numpy()
    # Concatenate directory and file name
    full_path_main_data = os.path.join(address, main_name)
    df1=df.to_pandas()
    #NVCOMP is every more compresssed
    df1.to_parquet(full_path_main_data,compression='zstd')  
    # np.save(full_path_main_data, data_array, allow_pickle=True)
    # np.save(full_path_header_data, headers_array, allow_pickle=True)

    # np.save(full_path_main_data, data_array)
    # np.save(full_path_header_data, headers_array)
    
def load_npy_CUdf(directory, filename):


    # Attempt to encode the first row
    # df.iloc[0, :] = df.iloc[0, :].apply(lambda x: x.encode('utf-8') if isinstance(x, str) else x)
    # # Attempt to encode the first column (Index), applying encoding only to strings
    #transforming all the kmers to ingegers
    #df.iloc[0:, 0] = df.iloc[0:, 0].apply(lambda x: x.encode('utf-8') if isinstance(x, str) else x)

    full_path_main_data = os.path.join(directory, filename)

    # # Load the numpy file
    # loaded_data = np.load(full_path_main_data)

    # # Create a cuDF DataFrame
    # df = cudf.DataFrame(loaded_data)

    df = cudf.read_parquet(full_path_main_data)
    return df




def save_cudf_csv(df,address,main_name):

    # Attempt to encode the first row
    # df.iloc[0, :] = df.iloc[0, :].apply(lambda x: x.encode('utf-8') if isinstance(x, str) else x)
    # # Attempt to encode the first column (Index), applying encoding only to strings
    #transforming all the kmers to ingegers
    #df.iloc[0:, 0] = df.iloc[0:, 0].apply(lambda x: x.encode('utf-8') if isinstance(x, str) else x)
    #data_array = df.to_pandas().to_numpy()
    # Concatenate directory and file name
    full_path_main_data = os.path.join(address, main_name)
    df1=df.to_pandas()
    #NVCOMP is every more compresssed
    df1.to_parquet(full_path_main_data,compression='zstd')  
    # np.save(full_path_main_data, data_array, allow_pickle=True)
    # np.save(full_path_header_data, headers_array, allow_pickle=True)

    # np.save(full_path_main_data, data_array)
    # np.save(full_path_header_data, headers_array)
    
def load_cudf_csv(directory, filename):
    

    # Attempt to encode the first row
    # df.iloc[0, :] = df.iloc[0, :].apply(lambda x: x.encode('utf-8') if isinstance(x, str) else x)
    # # Attempt to encode the first column (Index), applying encoding only to strings
    #transforming all the kmers to ingegers
    #df.iloc[0:, 0] = df.iloc[0:, 0].apply(lambda x: x.encode('utf-8') if isinstance(x, str) else x)

    full_path_main_data = os.path.join(directory, filename)

    # # Load the numpy file
    # loaded_data = np.load(full_path_main_data)

    # # Create a cuDF DataFrame
    # df = cudf.DataFrame(loaded_data)

    df = cudf.read_parquet(full_path_main_data)
    return df
