#!/usr/bin/env python3
import cudf
from src.dataframe_merge.dataframe_merge import dataframe_merge
from src.dataframe_merge.dataframe_merge import fill_NA_zero

def dataframe_merge_test():
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
