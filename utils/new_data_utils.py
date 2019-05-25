import numpy as np
from numpy.linalg import cholesky
import itertools
import random as my_random
import params_utils
import math_utils
import random

from PIL import Image
from pylab import *


class DataSet:
    fall_into_list_train = None
    fall_into_list_test = None


def get_stimulate_mu_cov():
    M = int(params_utils.Params.M)
    N = int(params_utils.Params.N)

    im = array(Image.open('resources/origin_data.jpg').convert('L'), 'f')
    M_0 = np.size(im, 0)
    sample_matrix = np.zeros((M, M))
    sample_mu_list = np.zeros(N)
    sample_data_table = np.zeros((params_utils.Params.N, 4), dtype=float)
    ij = itertools.product(range(M), range(M))

    for k, (i, j) in enumerate(ij):
        sample_matrix[i][j] = im[i * int(M_0 / M)][j * int(M_0 / M)]
        sample_data_table[k][0] = k
        sample_data_table[k][1] = i * (1.0 / params_utils.Params.M) + (0.5 / params_utils.Params.M)
        sample_data_table[k][2] = j * (1.0 / params_utils.Params.M) + (0.5 / params_utils.Params.M)
        sample_data_table[k][3] = sample_matrix[i][j]
        sample_mu_list[k] = sample_matrix[i][j]

    v_matrix = math_utils.get_v_matrix(sample_data_table[:, 1], sample_data_table[:, 2], params_utils.Params.N)
    return sample_mu_list, v_matrix


def generate_data_table(mu, cov):
    M = int(params_utils.Params.M)

    np.random.seed(1)
    rand_matrix = np.random.multivariate_normal(mean=mu, cov=cov)
    rand_matrix = rand_matrix.reshape((params_utils.Params.M, params_utils.Params.M))

    plt.matshow(rand_matrix, cmap='Greys_r')
    plt.savefig("output/rand_matrix.png")

    data_table = np.zeros((params_utils.Params.N, 4), dtype=float)
    ij = itertools.product(range(M), range(M))

    for k, (i, j) in enumerate(ij):
        data_table[k][0] = k
        data_table[k][1] = i * (1.0 / params_utils.Params.M) + (0.5 / params_utils.Params.M)
        data_table[k][2] = j * (1.0 / params_utils.Params.M) + (0.5 / params_utils.Params.M)
        data_table[k][3] = rand_matrix[i][j]

    return data_table


def init_data():
    print 'initializing data....'
    mu, cov = get_stimulate_mu_cov()
    data_table = generate_data_table(mu, cov)

    my_random.seed(2)
    training_slice = np.sort(my_random.sample(range(0, params_utils.Params.N),
                                                     int(params_utils.Params.N *
                                                         params_utils.Params.training_testing_split_ratio)))
    testing_slice = np.sort(np.setdiff1d(range(0, params_utils.Params.N), training_slice))

    DataSet.fall_into_list_train = data_table[training_slice,]
    DataSet.fall_into_list_test = data_table[testing_slice,]


def get_training_set():
    if DataSet.fall_into_list_train is None:
        init_data()
    return DataSet.fall_into_list_train


def get_testing_set():
    if DataSet.fall_into_list_test is None:
        init_data()
    return DataSet.fall_into_list_test
