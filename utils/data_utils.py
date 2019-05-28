import random as my_random

from PIL import Image
from pylab import *
from sklearn.model_selection import KFold

import utils.math_utils as math_utils
import utils.params_utils as params_utils


class DataSet:
    initialized = False
    train_list = []
    test_list = []


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
    print('initializing data....')
    DataSet.initialized = True

    mu, cov = get_stimulate_mu_cov()
    data_table = generate_data_table(mu, cov)

    my_random.seed(2)

    kf = KFold(n_splits=params_utils.Params.k_fold_splits, shuffle=True, random_state=0)
    for train_index, test_index in kf.split(data_table):
        if params_utils.Params.k_fold_train_by_small:
            DataSet.train_list.append(data_table[test_index,])
            DataSet.test_list.append(data_table[train_index,])
        else:
            DataSet.train_list.append(data_table[train_index,])
            DataSet.test_list.append(data_table[test_index,])

    # print("train_list: ", DataSet.train_list)
    # print("test_list: ", DataSet.test_list)


def get_training_sets():
    if DataSet.initialized is False:
        init_data()
    return DataSet.train_list


def get_testing_sets():
    if DataSet.initialized is False:
        init_data()
    return DataSet.test_list
