#!/usr/bin/env python3
import cudf
from src.config.config import Config


def dataframe_kmer_refrence_merge(refrence_kmer_dataframe,new_dataframe,name):
    print("Adding:")
    print(name)

    # Merge dataframe with df_union
    merged_dataframe = cudf.merge(refrence_kmer_dataframe, new_dataframe, on='K-mer', how='left')
    merged_dataframe.rename(columns={'Frequency':f'{name}'}, inplace=True)
    merged_dataframe[[name]]=(merged_dataframe[[name]]).astype('int32')
    return(merged_dataframe)



def fill_NA_zero(df):
    return(df.fillna(0))
    


