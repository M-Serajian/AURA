#!/usr/bin/env python3
import cudf


def dataframe_merge(union_dataframe,new_dataframe,name):

    # Merge dataframe with df_union
    union_dataframe = cudf.merge(union_dataframe, new_dataframe, on='K-mer', how='left')
    union_dataframe.rename(columns={'frequency':f'{name}'}, inplace=True)
    
    return(union_dataframe)


def fill_NA_zero(df):
    
    return(df.fillna(0))
    
