#!/usr/bin/env python3
import cudf
import pandas as pd
from src.dataframe_merge.dataframe_merge import dataframe_kmer_refrence_merge
from src.dataframe_merge.dataframe_merge import fill_NA_zero
from src.dataframe_npy_trasformation.dataframe_npy_trasformation import save_df_as_npy
import os
import copy
import numpy as np

def dataframe_merge_custom_squence_test():
    print("Testing dataframe merge function of CUDF Nvidia:")
    # Example data for df_union
    union_data = {'K-mer': ['AAA', 'CCC', 'GGG', 'TTT']}
    df_union = cudf.DataFrame(union_data)
    print("This is the main dataframe containing union of all the K-mers of the genomes")
    print(df_union)
    # Example data for df1, df2, and df3
    
    df1_data = {'K-mer': ['AAA', 'CCC', 'ACGT', 'GACT'],
                'frequency': [10, 20, 30, 40]}
    df1 = cudf.DataFrame(df1_data)
    print("This is df1 with frequencies for genome1")
    print(df1)

    df2_data = {'K-mer': ['AAA', 'GGG', 'CAGT', 'CCC'],
                'frequency': [15, 25, 35, 45]}
    df2 = cudf.DataFrame(df2_data)
    print("This is df2 with frequencies for genome2")
    print(df2)

    df3_data = {'K-mer': ['CCC', 'ACGT', 'GTCA', 'AAA'],
                'frequency': [18, 28, 38, 48]}
    df3 = cudf.DataFrame(df3_data)
    print("This is df3 with frequencies for genome3")
    print(df3)

    # List of dataframes
    dfs = [df1, df2, df3]

    # Check if 'K-mer' column is present in all dataframes
    k=1
    for df in dfs:
        df_union=dataframe_merge(df_union,df,f"df{k} K-mer frequency")
        k=k+1


    # Fill missing frequencies with 0
    df_union=fill_NA_zero(df_union)

    print("The output of the merge dataframe test is")
    print(df_union)
    print("------------------------------------------")



def dataframe_merge_CRyPTIC_test(number_of_samples,base,input_list,source_dataframe_dir):
    print(f"Testing dataframe merge function of CUDF Nvidia on {number_of_samples} genomes of CRyPTIC MTB data:",flush=True)
    # Example data for df_union ( refrence of all the unique kmers accross all the Gerbil outputs)
    source_dataframe = cudf.read_csv(source_dataframe_dir)
    source_dataframe = source_dataframe[["K-mer"]]
    print("Shape of source dataframe:",flush=True)
    print(source_dataframe.shape,flush=True)
    print("-------------",flush=True)



    union_dataframe= copy.copy(source_dataframe)
    #refrence_kmers = copy.copy(source_dataframe)
    
    print("DataFrame shape:", union_dataframe.shape)




    base_directory="/home/m.serajian/share/MTB/gerbil_output/csv/"
    
    
    # Threshold for dataframe size in bytes (60 GB)
    threshold = 60 * 1024 * 1024 * 1024  

    # Read the contents of the text file
    # with open("/home/m.serajian/projects/MTB_Plus_plus_GPU/tests/dataframe_merge/in_file_list.txt", 'r') as file:
    #     directories = file.readlines()
    # directories = [line.strip() for line in directories if line.strip()]
    # total_lines = len(directories)
    # Loop through directoriesinput_list


    with open(input_list, 'r') as file:
        directories = []
        for i, line in enumerate(file):
            if i >= number_of_samples:  # Stop reading after the first 100 lines
                break
            line = line.strip()
            if line:  # Only add non-empty lines
                directories.append(line)

    total_lines = len(directories)

    start_file_id=1
    end_file_id=0
    for idx, directory in enumerate(directories, start=1):
        print(f"idx= {idx}",flush=True)
        end_file_id=idx
        # Extract the parent directory containing "ERR"
        parent_directory = os.path.basename(os.path.dirname(directory))
        mid_df_dir = base_directory + parent_directory + ".csv"
        mid_dataframe = cudf.read_csv(mid_df_dir)
        
        # Merge dataframes
        mapped_dataframe = dataframe_kmer_refrence_merge(source_dataframe, mid_dataframe, parent_directory)

        union_dataframe = cudf.concat([union_dataframe, mapped_dataframe], axis=1)
        print(union_dataframe.memory_usage(deep=True).sum())
        if union_dataframe is None:
            print("Error: The DataFrame 'union_dataframe' is None.")
        else:
            print(union_dataframe.shape)
    # Optionally, add more debug information here to help trace the issue
        #print(union_dataframe.shape)
        if union_dataframe.memory_usage(deep=True).sum() > threshold:
    
            # Save and reset dataframe
            print("************************************************************",flush=True)
            print(idx)
            #union_dataframe_to_be_saved=union_dataframe.to_pandas()
 

            if union_dataframe is None:
                raise ValueError("Error: The DataFrame 'union_dataframe' is None.")
                # Optionally, add more debug information here to help trace the issue
            else:
                full_path_main_data = os.path.join(base, f"{start_file_id}_{end_file_id}.csv")
                #NVCOMP is every more compresssed
                union_dataframe.to_csv(full_path_main_data) 


            #del(union_dataframe)
            #union_dataframe_to_be_saved.to_csv(base_directory+f"{start_file_id}_{end_file_id}.csv")
            union_dataframe= copy.copy(source_dataframe)  
            start_file_id= end_file_id+1
            print("next batch being started!")
    
    if (start_file_id <= end_file_id): 

        if union_dataframe is None:
            raise ValueError("Error: The DataFrame 'union_dataframe' is None.")
            # Optionally, add more debug information here to help trace the issue
        else:
            full_path_main_data = os.path.join(base, f"{start_file_id}_{end_file_id}.csv")
            #NVCOMP is every more compresssed
            union_dataframe.to_csv(full_path_main_data) 

    print("Finished",flush=True)
    







