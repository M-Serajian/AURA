import cudf
import cupy as cp

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



df = cudf.read_csv('/home/m.serajian/projects/MTB_Plus_plus_GPU/data/Phenotypes/CRyPTIC_Phenotypes.csv')
INH=df['INH']
a,b=remove_ambiguous_phenotype_isolates(INH)
from cuml.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(a, b, test_size=0.2, random_state=42)

print(X_train)
print(X_test)
print(y_train)
print(y_test)

