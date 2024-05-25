#!/usr/bin/env python3
import cudf
from src.dataframe_merge.dataframe_merge import dataframe_merge
from src.dataframe_merge.dataframe_merge import fill_NA_zero
import os
import copy
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



def dataframe_merge_CRyPTIC_test():
    print("Testing dataframe merge function of CUDF Nvidia on 100 genomes of CRyPTIC MTB data:",flush=True)
    # Example data for df_union ( refrence of all the unique kmers accross all the Gerbil outputs)
    source_dataframe = cudf.read_csv("/home/m.serajian/share/MTB/gerbil_output/csv/1_1285_MTB_genomes.csv")
    print(source_dataframe.head(100))
    source_dataframe = source_dataframe[["K-mer"]]
    

    union_dataframe= copy.copy(source_dataframe)
    base_directory="/home/m.serajian/share/MTB/gerbil_output/csv/"
    
    
    # Threshold for dataframe size in bytes (60 GB)
    threshold = 60 * 1024 * 1024 * 1024  

    # Read the contents of the text file
    with open("/home/m.serajian/projects/MTB_Plus_plus_GPU/tests/dataframe_merge/in_file_list.txt", 'r') as file:
        directories = file.readlines()

    print("ok!")

    # Loop through directories
    iteration = 1
    for idx, directory in enumerate(directories, start=1):
        # Extract the parent directory containing "ERR"
        parent_directory = os.path.basename(os.path.dirname(directory))
        mid_df_dir = base_directory + parent_directory + ".csv"
        mid_dataframe = cudf.read_csv(mid_df_dir)
        print("ok!")

        # Merge dataframes
        union_dataframe = dataframe_merge(union_dataframe, mid_dataframe, parent_directory)

        # Check dataframe size every 200 iterations or at the end
        if idx % 3000 == 0 or idx == len(directories):
            # Check dataframe size
            if union_dataframe.memory_usage(deep=True).sum() > threshold:
                # Save and reset dataframe
                union_dataframe.to_csv(base_directory+f"final_merge_{iteration}.csv")
                union_dataframe= copy.copy(source_dataframe)  
                iteration += 1
            else:
                print(f"Iteration {idx}: Dataframe size is within threshold")

        union_dataframe.to_csv(base_directory+f"1_100_{iteration}.csv",index=False)
    print("------------------------------------------")
    

