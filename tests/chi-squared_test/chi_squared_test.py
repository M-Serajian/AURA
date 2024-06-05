
import cudf
import cupy as cp

# Example dataframes
df_labels = cudf.DataFrame({'id': [1, 2, 3, 4], 'label': ['A', 'B', 'A', 'B']})
df_observations = cudf.DataFrame({'id': [1, 2, 3, 4], 'observation': ['X', 'X', 'Y', 'Y']})

# Merge dataframes based on 'id' or appropriate key
df_merged = cudf.merge(df_labels, df_observations, on='id')

# Create contingency table
contingency_table = df_merged.groupby(['label', 'observation']).size().unstack(fill_value=0)

# Convert the contingency table to a CuPy array for calculation
observed = cp.array(contingency_table.values)

# Compute the expected frequencies
row_sums = cp.sum(observed, axis=1, keepdims=True)
col_sums = cp.sum(observed, axis=0, keepdims=True)
total = cp.sum(observed)
expected = row_sums * col_sums / total

# Calculate the Chi-squared statistic
chi_squared_stat = cp.sum((observed - expected) ** 2 / expected)

# Degrees of freedom
num_rows, num_cols = observed.shape
degrees_of_freedom = (num_rows - 1) * (num_cols - 1)

# Compute the p-value using the regularized gamma function as an approximation
p_value = cp.special.gammaincc(degrees_of_freedom / 2, chi_squared_stat / 2)
print(f"Chi-squared Statistic: {chi_squared_stat}, p-value: {p_value}")



def benjamini_hochberg(p_values, alpha=0.05):
    """
    Perform the Benjamini-Hochberg correction for multiple hypothesis testing on GPU.

    Parameters:
    p_values (cuPy array): Array of p-values to adjust.
    alpha (float): Significance level.

    Returns:
    cp.array: Adjusted p-values.
    """
    n = len(p_values)
    sorted_indices = cp.argsort(p_values)
    sorted_p_values = p_values[sorted_indices]

    # Calculate the cumulative minimum of the adjusted p-values
    adjusted_p_values = cp.empty(n, dtype=cp.float64)
    adjusted_p_values[sorted_indices] = n / (cp.arange(n) + 1) * sorted_p_values / alpha

    # Ensure that the adjusted p-values do not exceed 1
    adjusted_p_values = cp.minimum(adjusted_p_values, 1)

    return adjusted_p_values

# Example usage
p_values = cp.array([0.01, 0.04, 0.03, 0.20, 0.02])
adjusted_p_values = benjamini_hochberg(p_values)
print("Adjusted P-values:", adjusted_p_values.get())




# Test: scipy.stats vs GPU

