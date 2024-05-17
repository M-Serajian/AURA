#!/usr/bin/env python3


#rapids libs
import cupy as cp
import cudf


from src.config.config import Config

# Checking configuration
config = Config.get_instance()


def hashing(df):

    df["hash_md5"]=df.hash_values(method="md5")

    return(df)