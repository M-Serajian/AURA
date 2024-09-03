import cupy as cp
import cudf
from cupyx.scipy.special import gammainc
import pandas as pd
from scipy.stats import chi2_contingency 

# Step 1: Create larger datasets (size 100)
cp.random.seed(42)  # For reproducibility
array1 = cp.random.randint(0, 10, size=10000)  # Random non-negative integers
array2 = cp.random.randint(0, 2, size=10000)   # Random binary values


df1 = cudf.DataFrame({'Array1': array1})
df2 = cudf.DataFrame({'Array2': array2})


import cudf
import cupy as cp
from scipy.special import gammainc

def manual_chi2_test_cudf(array1, array2):
    # Ensure input arrays are cuDF Series
    array1 = cudf.Series(array1)
    array2 = cudf.Series(array2)
    
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


# def manual_chi2_test_cudf(df1, df2):
#     # Combine the two DataFrames into one
#     data = cudf.concat([df1, df2], axis=1)
#     data.columns = ['Array1', 'Array2']
    
#     # Create a contingency table using cuDF operations
#     contingency_table = data.groupby(['Array1', 'Array2']).size().reset_index(name='counts')
#     contingency_table = contingency_table.pivot(index='Array1', columns='Array2', values='counts').fillna(0)
    
#     # Calculate the row totals and column totals using cuDF
#     row_totals = contingency_table.sum(axis=1)
#     col_totals = contingency_table.sum(axis=0)
#     total = contingency_table.sum().sum()

#     # Calculate expected frequencies using CuPy (GPU)
#     row_totals_cp = row_totals.values
#     col_totals_cp = col_totals.values
#     expected = cudf.DataFrame(cp.outer(row_totals_cp, col_totals_cp) / total,
#                               index=contingency_table.index, columns=contingency_table.columns)
    
#     # Calculate the chi-square statistic manually using GPU
#     observed = contingency_table
#     chi_square_stat = ((observed - expected) ** 2 / expected).sum().sum()
    
#     # Degrees of freedom
#     dof = (observed.shape[0] - 1) * (observed.shape[1] - 1)
    
#     # Manual p-value calculation using the chi-square distribution on GPU
#     p_value = 1 - gammainc(dof / 2.0, chi_square_stat / 2.0)

#     return chi_square_stat, p_value, dof


score,p,doff=manual_chi2_test_cudf(array1,array2)

print(score)
print(p)
print(doff)
print("-----------")
data = cudf.concat([df1, df2], axis=1)

# Group by Array1 and Array2, and then count occurrences
contingency_table = data.groupby(['Array1', 'Array2']).size().reset_index(name='counts')

# Pivot the table to create a contingency table
contingency_table = contingency_table.pivot(index='Array1', columns='Array2', values='counts').fillna(0)

# Convert the resulting cuDF DataFrame to a NumPy array for chi2_contingency
contingency_table_np = contingency_table.to_pandas().values

# Perform the chi-square test for independence using chi2_contingency
chi2_stat, p_value, dof, expected = chi2_contingency(contingency_table_np)

print(f"Chi-square Statistic: {chi2_stat}")
print(f"p-value: {p_value}")
print(f"Degrees of Freedom: {dof}")
