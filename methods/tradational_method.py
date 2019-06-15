import numpy as npfrom utils import math_utils# y_hat = E(y(s)) + K^T V^{-1} (y-E(y))def test_model(params, x, y, z, n, v_matrix, mean, testing_set):    sst = np.sum((testing_set[:, 3] - np.mean(testing_set[:, 3])) ** 2)    sse = 0    for i in range(0, np.size(testing_set, 0)):        k_vector = math_utils.get_k_vector_for(params, x, y, n, testing_set[i, 1], testing_set[i, 2])        y_hat = mean + np.matmul(np.matmul(k_vector.T, np.linalg.inv(v_matrix)), z)        # print('Y_hat={}, Y real value=={}'.format(y_hat[0], testing_set[i, 3]))        residual = testing_set[i, 3] - y_hat[0]        # print('residual=={}'.format(residual))        sse += residual ** 2    r_2 = 1 - sse / sst    return r_2def traditional_method(training_set, testing_set, params):    x = training_set[:, 1]    y = training_set[:, 2]    z = training_set[:, 3]    n = np.size(training_set, 0)    mean = np.mean(z)    z = z - mean    v_matrix = math_utils.get_v_matrix(params, x, y, n)    cond_number = np.linalg.cond(v_matrix)    print('Conditional number is: {}'.format(cond_number.__str__()))    return test_model(params, x, y, z, n, v_matrix, mean, testing_set), cond_number