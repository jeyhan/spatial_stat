import numpy as np


def spatial_cov(params, v1x, v1y, v2x, v2y):
    dist = np.sqrt((v1x - v2x) ** 2 + (v1y - v2y) ** 2)
    #    exponential_variance = params_utils.once_differentiable_matern_covariance_function(dist)
    #    exponential_variance = params_utils.once_differentiable_matern_covariance_function(dist)
    exponential_variance = params.once_differentiable_matern_covariance_function(dist)
    return exponential_variance


def get_v_matrix(params, x, y, n):
    v_matrix0 = np.zeros((n, n))
    for i in range(0, n):
        for j in range(0, n):
            v_matrix0[i][j] = spatial_cov(params, x[i], y[i], x[j], y[j])

    return v_matrix0


def get_k_vector_for(params, x, y, n, new_x, new_y):
    k_vector0 = np.zeros((n, 1))
    for i in range(0, n, 1):
        k_vector0[i][0] = spatial_cov(params, x[i], y[i], new_x, new_y)

    return k_vector0
