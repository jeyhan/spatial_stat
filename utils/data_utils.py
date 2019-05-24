import numpy as np
from numpy.linalg import cholesky
import matplotlib.pyplot as plt
import itertools
import random
import params_utils
import math_utils



class DataSet:
    fall_into_list_train = None
    fall_into_list_test = None


def get_2norm_fall_into_fine_grid_table():
    sample_no = params_utils.Params.data_sample_count
    var = params_utils.Params.data_var
    np.random.seed(0)
    # generate many points from 2-dim independent normal distribution.
    mu = np.array([[0, 0]])
    sigma = np.array([[var, 0], [0, var]])
    r = cholesky(sigma)
    n2_data = np.abs(np.dot(np.random.randn(sample_no, 2), r) + mu)
    fall_into_matrix = np.zeros((params_utils.Params.M, params_utils.Params.M), dtype=int)
    # obs, x-position, y-position, count at there.
    fall_into_list1 = np.zeros((params_utils.Params.N, 4), dtype=float)

    for i in range(0, sample_no):
        if (n2_data[i][0] > 0) and (n2_data[i][0] <= 1) and (n2_data[i][1] > 0) and (n2_data[i][1] <= 1):
            fall_into_matrix[int(n2_data[i][0] * params_utils.Params.M)][int(n2_data[i][1] * params_utils.Params.M)] += 1
            fall_into_matrix[int(n2_data[i][0] * params_utils.Params.M)][int(n2_data[i][1] * params_utils.Params.M)] += 1

    ij = itertools.product(range(params_utils.Params.M), range(params_utils.Params.M))
    for k, (i, j) in enumerate(ij):
        fall_into_list1[k][0] = k
        fall_into_list1[k][1] = i * (1.0 / params_utils.Params.M) + (0.5 / params_utils.Params.M)
        fall_into_list1[k][2] = j * (1.0 / params_utils.Params.M) + (0.5 / params_utils.Params.M)
        fall_into_list1[k][3] = fall_into_matrix[i][j]

    return fall_into_list1



def init_data():
    print 'initializing data....'
    fall_into_list = get_2norm_fall_into_fine_grid_table()

    # print(fall_into_list)

    if params_utils.Params.M <= 30:
        fig = plt.figure(figsize=(6, 6))
        ax = plt.subplot()
        ax.scatter(fall_into_list[:, 1],
                   fall_into_list[:, 2],
                   c=fall_into_list[:, 3],
                   s=100000 / params_utils.Params.N, marker='s')
        plt.axis([0, 1, 0, 1])
        plt.title('[N(0, 0.49), N(0, 0.49)] fall in fine grid')
        fig.savefig("output/fine_grid_fall.png")

    v_mat = np.zeros((params_utils.Params.N, params_utils.Params.N))

    ij = itertools.product(range(params_utils.Params.N), range(params_utils.Params.N))
    for k, (i, j) in enumerate(ij):
        v_mat[i][j] = math_utils.spatial_cov(fall_into_list[i][1],
                                  fall_into_list[i][2],
                                  fall_into_list[j][1],
                                  fall_into_list[j][2])
    v_mat_cond_number = np.linalg.cond(v_mat)
    # the cond number is 47035, it is a huge number

    # we split 10*10=100 points into -> m*m=n points.
    # we can use small set to predict large set.

    random.seed(2)

    training_slice = np.sort(random.sample(range(0, params_utils.Params.N),
                                           int(params_utils.Params.N * params_utils.Params.training_testing_split_ratio)))
    testing_slice = np.sort(np.setdiff1d(range(0, params_utils.Params.N), training_slice))

    DataSet.fall_into_list_train = fall_into_list[training_slice,]
    DataSet.fall_into_list_test = fall_into_list[testing_slice,]

    #    print DataSet.fall_into_list_train
    #    print DataSet.fall_into_list_test
    if params_utils.Params.M <= 30:
        fig = plt.figure(figsize=(6, 6))
        ax = plt.subplot()
        ax.scatter(DataSet.fall_into_list_train[:, 1],
                   DataSet.fall_into_list_train[:, 2],
                   c=DataSet.fall_into_list_train[:, 3],
                   s=100000 / params_utils.Params.N, marker='s')
        plt.axis([0, 1, 0, 1])
        plt.title('Sampled data from all (as training set)')
        fig.savefig("output/training_grid_fall.png")

    # then we can use it to predict testing value and compare it to real value.


def get_training_set():
    if DataSet.fall_into_list_train is None:
        init_data()
    return DataSet.fall_into_list_train


def get_testing_set():
    if DataSet.fall_into_list_test is None:
        init_data()
    return DataSet.fall_into_list_test



