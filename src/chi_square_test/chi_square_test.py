import cupy as cp
from cupyx.scipy.sparse import csr_matrix

import cupy as cp


# def chi_squared_test_yates_correction(labels: cp.ndarray, feature_matrix: cp.ndarray) -> cp.ndarray:
#     """
#     Perform the Chi-squared test with Yates' continuity correction for multiple features.

#     Parameters:
#         labels (cp.ndarray): Binary labels (0 or 1) of shape (n_samples,) or (1, n_samples).
#         feature_matrix (cp.ndarray): Binary feature matrix of shape (n_samples, n_features).

#     Returns:
#         cp.ndarray: Chi-squared scores for each feature.
#     """

#     yates_correction_constant=0.5

#     # Reshape labels to shape (1, n_samples) if necessary
#     if labels.ndim == 1 or labels.shape[0] != 1:
#         labels = labels.reshape(1, -1)

#     n_samples = labels.shape[1]
#     n_features = feature_matrix.shape[1]

#     # Calculate observed counts
#     count_label1_feature1 = labels @ feature_matrix  # Both label and feature are 1
#     count_label0_feature1 = (1 - labels) @ feature_matrix  # Label is 0, feature is 1
#     count_label1_feature0 = labels.sum() - count_label1_feature1  # Label is 1, feature is 0
#     count_label0_feature0 = (n_samples - labels.sum()) - count_label0_feature1  # Both label and feature are 0

#     print(count_label1_feature1[0,0])
#     print(count_label0_feature1[0,0])
#     print(count_label1_feature0[0,0])
#     print(count_label0_feature0[0,0])
#     print("------------------")
#     print(count_label1_feature1[0,1])
#     print(count_label0_feature1[0,1])
#     print(count_label1_feature0[0,1])
#     print(count_label0_feature0[0,1])
#     print("------------------")
#     print(count_label1_feature1[0,2])
#     print(count_label0_feature1[0,2])
#     print(count_label1_feature0[0,2])
#     print(count_label0_feature0[0,2])

#     # Row and column totals
#     total_labels_0 = count_label0_feature0 + count_label0_feature1
#     total_labels_1 = count_label1_feature0 + count_label1_feature1
#     total_features_0 = count_label0_feature0 + count_label1_feature0
#     total_features_1 = count_label0_feature1 + count_label1_feature1
#     grand_total = n_samples * cp.ones((1, n_features), dtype=cp.float32)

#     # Calculate expected frequencies
#     expected_label0_feature0 = (total_labels_0 * total_features_0) / grand_total
#     expected_label0_feature1 = (total_labels_0 * total_features_1) / grand_total
#     expected_label1_feature0 = (total_labels_1 * total_features_0) / grand_total
#     expected_label1_feature1 = (total_labels_1 * total_features_1) / grand_total

#     # Avoid division by zero
#     epsilon = 1e-10
#     expected_label0_feature0 = cp.maximum(expected_label0_feature0, epsilon)
#     expected_label0_feature1 = cp.maximum(expected_label0_feature1, epsilon)
#     expected_label1_feature0 = cp.maximum(expected_label1_feature0, epsilon)
#     expected_label1_feature1 = cp.maximum(expected_label1_feature1, epsilon)

#     # Calculate Chi-squared statistic with Yates' continuity correction
#     chi2_scores = (
#         ((cp.abs(count_label0_feature0 - expected_label0_feature0) - yates_correction_constant) ** 2) / expected_label0_feature0 +
#         ((cp.abs(count_label0_feature1 - expected_label0_feature1) - yates_correction_constant) ** 2) / expected_label0_feature1 +
#         ((cp.abs(count_label1_feature0 - expected_label1_feature0) - yates_correction_constant) ** 2) / expected_label1_feature0 +
#         ((cp.abs(count_label1_feature1 - expected_label1_feature1) - yates_correction_constant) ** 2) / expected_label1_feature1
#     )

#     return chi2_scores


import cupy as cp

def chi_squared_test_yates_correction(labels: cp.ndarray, feature_matrix: cp.ndarray) -> cp.ndarray:
    """
    Perform the Chi-squared test with Yates' continuity correction for multiple features.

    Parameters:
        labels (cp.ndarray): Binary labels (0 or 1) of shape (n_samples,) or (1, n_samples).
        feature_matrix (cp.ndarray): Binary feature matrix of shape (n_samples, n_features).

    Returns:
        cp.ndarray: Chi-squared scores for each feature.
    """

    yates_correction_constant = 0.5

    # Reshape labels to shape (1, n_samples) if necessary
    if labels.ndim == 1 or labels.shape[0] != 1:
        labels = labels.reshape(1, -1)

    labels = labels.astype(cp.float64)
    feature_matrix = feature_matrix.astype(cp.float64)

    n_samples = labels.shape[1]
    n_features = feature_matrix.shape[1]

    # Calculate observed counts
    count_label1_feature1 = labels @ feature_matrix  # Both label and feature are 1
    count_label0_feature1 = (1 - labels) @ feature_matrix  # Label is 0, feature is 1
    count_label1_feature0 = labels.sum() - count_label1_feature1  # Label is 1, feature is 0
    count_label0_feature0 = (n_samples - labels.sum()) - count_label0_feature1  # Both label and feature are 0

    # Row and column totals
    total_labels_0 = count_label0_feature0 + count_label0_feature1  # Total counts where label is 0
    total_labels_1 = count_label1_feature0 + count_label1_feature1  # Total counts where label is 1
    total_features_0 = count_label0_feature0 + count_label1_feature0  # Total counts where feature is 0
    total_features_1 = count_label0_feature1 + count_label1_feature1  # Total counts where feature is 1
    grand_total = n_samples * cp.ones((1, n_features), dtype=cp.float64)  # Total sample size for each feature

    # Calculate expected frequencies
    expected_label0_feature0 = (total_labels_0 * total_features_0) / grand_total
    expected_label0_feature1 = (total_labels_0 * total_features_1) / grand_total
    expected_label1_feature0 = (total_labels_1 * total_features_0) / grand_total
    expected_label1_feature1 = (total_labels_1 * total_features_1) / grand_total

    # Avoid division by zero by setting a small epsilon value
    epsilon = 1e-10
    expected_label0_feature0 = cp.maximum(expected_label0_feature0, epsilon)
    expected_label0_feature1 = cp.maximum(expected_label0_feature1, epsilon)
    expected_label1_feature0 = cp.maximum(expected_label1_feature0, epsilon)
    expected_label1_feature1 = cp.maximum(expected_label1_feature1, epsilon)

    # Calculate adjusted differences and apply Yates' correction
    adj_diff_label0_feature0 = cp.maximum(cp.abs(count_label0_feature0 - expected_label0_feature0) - yates_correction_constant, 0)
    adj_diff_label0_feature1 = cp.maximum(cp.abs(count_label0_feature1 - expected_label0_feature1) - yates_correction_constant, 0)
    adj_diff_label1_feature0 = cp.maximum(cp.abs(count_label1_feature0 - expected_label1_feature0) - yates_correction_constant, 0)
    adj_diff_label1_feature1 = cp.maximum(cp.abs(count_label1_feature1 - expected_label1_feature1) - yates_correction_constant, 0)

    # Calculate Chi-squared statistic with corrected adjusted differences
    chi2_scores = (
        (adj_diff_label0_feature0 ** 2) / expected_label0_feature0 +
        (adj_diff_label0_feature1 ** 2) / expected_label0_feature1 +
        (adj_diff_label1_feature0 ** 2) / expected_label1_feature0 +
        (adj_diff_label1_feature1 ** 2) / expected_label1_feature1
    )

    return chi2_scores
