#!/usr/bin/env python3


#system libs to make full package 
import sys
import os

# Calculate the absolute path to the directory where main.py resides
# This assumes main.py is in the root of your project
project_root = os.path.abspath(os.path.dirname(__file__))
# Add the project root to the Python path
sys.path.insert(0, project_root)




#python libs
import numpy as np
import argparse


#rapids libs
import cupy as cp
import cudf

#rapids memory management
import rmm

# Create a Device object for the default GPU device
device = cp.cuda.Device(0)

# Get the total memory of the GPU
total_gpu_memory = device.mem_info[1]
print(f"Total VRAM is :{total_gpu_memory}")
# Calculate a percentage of the total memory to use for the RMM pool (e.g., 80%)
# and ensure it is a multiple of 256 bytes
pool_size = int(total_gpu_memory * 0.8) // 256 * 256

# Reinitialize RMM with the calculated pool size
rmm.reinitialize(
    pool_allocator=True,  # Enable the memory pool
    initial_pool_size=pool_size  # Set the pool size
)


#Recommended by Oded : 
# import rmm
# pool = rmm.mr.PoolMemoryResource(rmm.mr.CudaMemoryResource(),\
#                                 initial_pool_size=2**30,\
#                                 maximum_pool_size=2**32)
# rmm.mr.set_current_device_resource(pool)



#Configuration modules
from src.arg_pars.argument_pars import parse_arguments
from src.config.config import Config


# Setup argparse
args = parse_arguments()


# Set configuration
config = Config.get_instance()
config.set_debug(args.debug)
config.set_test(args.test)



# manual libraries after configurarion is complete 
from src.hashing.hash import hashing


# manual libraries after configurarion for testing 
from tests.collision.collision_detector import detect_collision_test
from tests.dataframe_merge.dataframe_merge_test import dataframe_merge_custom_squence_test
from tests.dataframe_merge.dataframe_merge_test import dataframe_merge_CRyPTIC_test
from src.dataframe_npy_trasformation.dataframe_npy_trasformation import load_npy_CUdf







def main():

    # Runing tests if flag is 1
    if config.test:
        print("Test mode:")

        #detect_collision_test("/home/m.serajian/share/MTB/gerbil_output/csv/k-mer_samples.csv")
        base="/home/m.serajian/share/MTB/gerbil_output/csv/"
        source_dataframe_dir="/home/m.serajian/share/MTB/gerbil_output/csv/1_1285_MTB_genomes.csv"
        input_list="/home/m.serajian/projects/MTB_Plus_plus_GPU/tests/dataframe_merge/in_file_list.txt"
        dataframe_merge_CRyPTIC_test(100,base,input_list,source_dataframe_dir)






if __name__ == "__main__":
    main()