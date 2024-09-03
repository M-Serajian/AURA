#!/usr/bin/env python3


#system libs to make full package 
import sys
import os


# profiling
import cProfile
import pstats

# Calculate the absolute path to the directory where main.py resides
# This assumes main.py is in the root of your project
project_root = os.path.abspath(os.path.dirname(__file__))
# Add the project root to the Python path
sys.path.insert(0, project_root)




#python libs
# import numpy as np
# import argparse


#rapids libs
import cupy as cp
import cudf
from cupyx.scipy.special import gammainc

from cuml.linear_model import LogisticRegression
from cuml.ensemble import RandomForestClassifier
from cuml.model_selection import train_test_split
from scipy.sparse import csr_matrix
import cupy as cp
from xgboost import XGBClassifier
from cuml.model_selection import train_test_split
from cuml.metrics import confusion_matrix
from cuml.linear_model import Lasso
#rapids memory management
import rmm

# Create a Device object for the default GPU device
device = cp.cuda.Device(0)

# Get the total memory of the GPU
total_gpu_memory = device.mem_info[1]
print(f"Total VRAM is :{total_gpu_memory}")
# Calculate a percentage of the total memory to use for the RMM pool (e.g., 80%)
# and ensure it is a multiple of 256 bytes
pool_size = int(total_gpu_memory * 0.6) // 256 * 256

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
print(args.input)


# Set configuration
config = Config.get_instance()
config.set_debug(args.debug)
config.set_test(args.test)
# config.load_dataframe(args.input)


# manual libraries after configurarion is complete 
from src.hashing.hash import hashing


# manual libraries after configurarion for testing 
from tests.collision.collision_detector import detect_collision_test
#from tests.dataframe_merge.dataframe_merge_test import dataframe_merge_custom_squence_test
from tests.dataframe_merge.dataframe_merge_test import dataframe_merge_CRyPTIC_test
from src.dataframe_npy_trasformation.dataframe_npy_trasformation import load_npy_CUdf







def manual_chi2_test_cudf(array1, array2):


    if (cp.all(array1 == 0) or cp.all(array1 == 1) or cp.all(array2 == 0) or cp.all(array2 == 1)):
        return 0,0,0
    else:
        # Create a contingency table using cuDF operations
        data = cudf.DataFrame({'Array1': array1, 'Array2': array2})
        contingency_table = data.groupby(['Array1', 'Array2']).size().reset_index(name='counts')
        contingency_table = contingency_table.pivot(index='Array1', columns='Array2', values='counts').fillna(0)
        
        # Calculate the row totals and column totals using cuDF
        row_totals = contingency_table.sum(axis=1)
        col_totals = contingency_table.sum(axis=0)
        total = contingency_table.sum().sum()

        # Calculate expected frequencies using cuDF
        row_totals_cp = row_totals.to_cupy()
        col_totals_cp = col_totals.to_cupy()
        expected = cudf.DataFrame(cp.outer(row_totals_cp, col_totals_cp) / total,
                                index=contingency_table.index, columns=contingency_table.columns)
        
        # Calculate the chi-square statistic manually using cuDF
        observed = contingency_table
        chi_square_stat = ((observed - expected) ** 2 / expected).sum().sum()
        
        # Degrees of freedom
        dof = (observed.shape[0] - 1) * (observed.shape[1] - 1)
        
        # Manual p-value calculation using the chi-square distribution on GPU
        p_value = 1 - cp.asarray(gammainc(dof / 2.0, chi_square_stat / 2.0)).item()

        return chi_square_stat, p_value, dof
    
    
def main():
    from src.sparse_matrix_representation.sparse_matrix_representation import dataframe_merge_CSR_parquet
    from src.sparse_matrix_representation.sparse_matrix_representation import dataframe_merge_CSR_csv
    from src.sparse_matrix_representation.sparse_matrix_representation import initial_dataframe_merge_CSR_csv
    from src.sparse_matrix_representation.sparse_matrix_representation import dataframe_merge_CSR_csv_input_list
    # Runing tests if flag is 1
    if config.test:
        print("Test mode:")

        
        source_dataframe_dir="/home/m.serajian/share/MTB/gerbil_output/csv/1_1285_MTB_genomes.csv"


        file_path = '/home/m.serajian/projects/MTB_Plus_plus_GPU/data/genome_directories/1000_cryptic_genomes.txt'

        # Function to extract parent directories
        def extract_parent_directories(file_path):
            parent_directories = []
            with open(file_path, 'r') as file:
                for line in file:
                    line = line.strip()
                    if line:  # Ensure the line is not empty
                        parent_directory = os.path.basename(os.path.dirname(line))
                        parent_directories.append(parent_directory)
            return parent_directories

        # Get the parent directories
        parent_directories = extract_parent_directories(file_path)

        gerbil_csv_DIRs=[]
        # Print the parent directories
        for directory in parent_directories:
            gerbil_csv_DIRs.append("/home/m.serajian/share/MTB/gerbil_output/csv/"+directory+".csv")


        matrix = initial_dataframe_merge_CSR_csv(gerbil_csv_DIRs,source_dataframe_dir)
      
        def remove_ambiguous_phenotype_isolates(phenotype):
            """
            Remove isolates with ambiguous (NaN) phenotype values and return their indices and values as arrays.

            Parameters:
            phenotype (cudf.Series): The phenotype Series.

            Returns:
            cp.ndarray: The indices of the non-NaN values.
            cp.ndarray: The non-NaN phenotype values as integers.
            """
            # Filter out the rows where the phenotype values are NaN
            non_nan_values = phenotype.dropna()

            # Convert the indices to a CuPy array
            non_nan_indices = cp.asarray(non_nan_values.index.values)

            # Convert the values to a CuPy array of integers
            non_nan_values_array = non_nan_values.values.astype(cp.int32)

            return non_nan_indices, non_nan_values_array



        phenotypes = cudf.read_csv('/home/m.serajian/projects/MTB_Plus_plus_GPU/data/Phenotypes/CRyPTIC_Phenotypes.csv')
        INH=phenotypes['INH']
        indecies,phenotypic_data=remove_ambiguous_phenotype_isolates(INH)
        
        indecies=indecies[0:1000]
        phenotypic_data=phenotypic_data[0:1000]

        
        train_indecies, test_indecies, train_phenotypic_data, test_phenotypic_data = train_test_split(indecies, phenotypic_data, test_size=0.2, random_state=42)
        train_indecies=train_indecies.flatten() 
        test_indecies =test_indecies.flatten() 
        chi_score_array = cp.zeros(matrix.shape[1])
        counter=0
        import time
        t0=time.time()
        for i in range(matrix.shape[1]):
            counter=counter+1
            data=(matrix[train_indecies,i]).toarray().ravel()
            score,_,_=manual_chi2_test_cudf(data,train_phenotypic_data)

            chi_score_array[i]=score
            if (counter==10000):
                print(time.time()-t0)
        
        top_k=2**14
        sorted_indices = cp.argsort(chi_score_array)[-top_k:][::-1]

        train_data=((matrix[train_indecies,:][:,sorted_indices]).toarray())


        test_data=((matrix[test_indecies,:][:,sorted_indices])).toarray()
        
        print("Train data shape:", train_data.shape,flush=True)
        print("Test data shape:", test_phenotypic_data.shape,flush=True)
        # xgb_model = XGBClassifier(tree_method='gpu_hist', use_label_encoder=False, eval_metric='logloss')

        # xgb_model.fit(train_data, train_phenotypic_data)  # y_train.get() to convert CuPy array to NumPy

        # # Predict on the test set
        # y_pred_xgb = xgb_model.predict(test_data)

        # cm = confusion_matrix(test_phenotypic_data, y_pred_xgb)
        # F1 score calculation
        model = LogisticRegression(penalty='l1', solver='qn', max_iter=1000)

        model.fit(train_data, train_phenotypic_data)
        y_pred = model.predict(test_data)
        cm = confusion_matrix(test_phenotypic_data, y_pred)

        tn, fp, fn, tp = cm.ravel()
        precision = tp / (tp + fp)
        recall = tp / (tp + fn)
        f1 = 2 * (precision * recall) / (precision + recall)
        print(f"LR Lasso F1 Score: {f1}")

        rf_model = RandomForestClassifier(n_estimators=100, max_depth=10)
        rf_model.fit(train_data, train_phenotypic_data)
        y_pred = (rf_model.predict(test_data)).astype(cp.int32)
        cm = confusion_matrix(test_phenotypic_data, y_pred)

        tn, fp, fn, tp = cm.ravel()
        precision = tp / (tp + fp)
        recall = tp / (tp + fn)
        f1 = 2 * (precision * recall) / (precision + recall)
        print(f"RF Score: {f1}")

        xgb_classifier = XGBClassifier(verbosity=1)
        xgb_classifier.fit(train_data, train_phenotypic_data)
        y_pred = xgb_classifier.predict(test_data)
        cm = confusion_matrix(test_phenotypic_data, y_pred)

        tn, fp, fn, tp = cm.ravel()
        precision = tp / (tp + fp)
        recall = tp / (tp + fn)
        f1 = 2 * (precision * recall) / (precision + recall)
        print(f"xgboost Score: {f1}")


if __name__ == "__main__":
    # Profile the main function
    # profiler = cProfile.Profile()
    # profiler.enable()
    main()
    # profiler.disable()
    # # Save the profiling results
    # profiler.dump_stats("profile_stats")

    # # Optionally, print the profiling results
    # with open("profile_stats_parquet.txt", "w") as f:
    #     ps = pstats.Stats(profiler, stream=f)
    #     ps.strip_dirs().sort_stats('cumulative').print_stats()