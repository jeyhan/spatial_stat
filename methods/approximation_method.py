import numpy as np

from utils import math_utils


# neighbor count or distance.
def get_neighbor_list_old(x, y, i, neighbor_range):
    neighbor_set = [i]
    for j in range(0, x.size, 1):
        if np.sqrt((x[j] - x[i]) ** 2 + (y[j] - y[i]) ** 2) < neighbor_range and i != j:
            neighbor_set.append(j)
    return neighbor_set


def get_neighbor_index_list(x, y, n, ii, neighbor_size):
    # i, dist
    neighbor_list = np.zeros((n, 2))
    for jj in range(0, n):
        dist = np.sqrt((x[jj] - x[ii]) ** 2 + (y[jj] - y[ii]) ** 2)
        neighbor_list[jj, 0] = dist
        neighbor_list[jj, 1] = jj

    index_sorted = np.argsort(neighbor_list[:, 0])
    neighbor_index_list = neighbor_list[index_sorted,][int(0):int(neighbor_size), 1]
    return neighbor_index_list.astype(np.int)


def predict_point_by_exclude_i(z, i, neighbor_index_list, v_matrix_inv):
    y_hat_i = z[i] - (1 / v_matrix_inv[0][0]) * np.sum(np.dot(z[neighbor_index_list,], v_matrix_inv[0, :]))
    return y_hat_i


def test_model(params, x, y, n, v_inv_y, mean, testing_set):
    sst = np.sum((testing_set[:, 3] - np.mean(testing_set[:, 3])) ** 2)

    sse = 0
    for i in range(0, np.size(testing_set, 0)):
        k_vector = math_utils.get_k_vector_for(params, x, y, n, testing_set[i, 1], testing_set[i, 2])
        y_hat = np.matmul(k_vector.T, v_inv_y) + mean
        # print('Y_hat={}, Y real value=={}'.format(y_hat[0], testing_set[i, 3]))
        residual = testing_set[i, 3] - y_hat[0]
        # print('residual=={}'.format(residual))
        sse += residual ** 2

    r_2 = 1 - sse / sst
    return r_2[0]


# v^{-1}y = diag(v^{-1})(y-y_hat)
def approximation_method(training_set, testing_set, params):
    x = training_set[:, 1]
    y = training_set[:, 2]
    z = training_set[:, 3]
    n = np.size(training_set, 0)
    mean = np.mean(z)
    z = z - mean

    v_inv_y = np.zeros((n, 1))

    cond_numbers = 0
    for i in range(0, n):
        neighbor_size = int(params.neighbor_relative_ratio * n)

        neighbor_index_list = get_neighbor_index_list(x, y, n, i, neighbor_size)

        v_matrix = math_utils.get_v_matrix(params, x[neighbor_index_list,], y[neighbor_index_list,], neighbor_size)
        cond_numbers += np.linalg.cond(v_matrix)

        v_matrix_inv = np.linalg.inv(v_matrix)
        pred_hat = predict_point_by_exclude_i(z, i, neighbor_index_list, v_matrix_inv)

        #        print('Y hat ={}'.format(pred_hat))
        #        print('Y real={}'.format(z[i]))

        #        print v_matrix_inv[0][0]

        v_inv_y[i][0] = v_matrix_inv[0][0] * (z[i] - pred_hat)

    return test_model(params, x, y, n, v_inv_y, mean, testing_set), (cond_numbers / n)
