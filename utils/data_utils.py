import itertools as itertool
import random as my_random

from pylab import *
from sklearn.model_selection import KFold

import utils.params_utils as params_utils


class DataSet:
    initialized = False
    train_list = []
    test_list = []


def generate_data_table(mu, cov):
    M = int(params_utils.Params.M)

    np.random.seed(1)

    stimulate_matrix = np.random.multivariate_normal(mean=mu, cov=cov)
    stimulate_matrix = stimulate_matrix.reshape((params_utils.Params.M, params_utils.Params.M))

    plt.matshow(stimulate_matrix, cmap='Greys_r')
    plt.savefig("output/stimulate_matrix.png")

    data_table = np.zeros((params_utils.Params.N, 4), dtype=float)
    ij = itertool.product(range(M), range(M))

    for k, (i, j) in enumerate(ij):
        data_table[k][0] = k
        data_table[k][1] = i * (1.0 / params_utils.Params.M) + (0.5 / params_utils.Params.M)
        data_table[k][2] = j * (1.0 / params_utils.Params.M) + (0.5 / params_utils.Params.M)
        data_table[k][3] = stimulate_matrix[i][j]

    return data_table


def init_data():
    print('initializing data....')
    DataSet.initialized = True

    mu, cov = params_utils.get_stimulate_mu_cov()
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
