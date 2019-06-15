import itertools as itertool
import random as my_random

from pylab import *
from sklearn.model_selection import KFold


class DataSet:
    params = None
    initialized = False
    train_list = None
    test_list = None

    def __init__(self, params):
        self.params = params
        self.initialized = False
        self.train_list = []
        self.test_list = []

        self.init_data()

    def init_data(self):
        print('initializing data by params....')
        self.initialized = True

        mu, cov = self.params.get_stimulate_mu_cov()
        data_table = self.generate_data_table(mu, cov)

        my_random.seed(2)

        kf = KFold(n_splits=self.params.k_fold_splits, shuffle=True, random_state=0)
        for train_index, test_index in kf.split(data_table):
            if self.params.k_fold_train_by_small:
                self.train_list.append(data_table[test_index,])
                self.test_list.append(data_table[train_index,])
            else:
                self.train_list.append(data_table[train_index,])
                self.test_list.append(data_table[test_index,])

        # print("train_list: ", self.train_list)
        # print("test_list: ", self.test_list)

    def generate_data_table(self, mu, cov):
        M = int(self.params.M)

        np.random.seed(1)

        stimulate_matrix = np.random.multivariate_normal(mean=mu, cov=cov)
        stimulate_matrix = stimulate_matrix.reshape((self.params.M, self.params.M))

        plt.matshow(stimulate_matrix, cmap='Greys_r')
        plt.savefig("output/stimulate_matrix.png")

        data_table = np.zeros((self.params.N, 4), dtype=float)
        ij = itertool.product(range(M), range(M))

        for k, (i, j) in enumerate(ij):
            data_table[k][0] = k
            data_table[k][1] = i * (1.0 / self.params.M) + (0.5 / self.params.M)
            data_table[k][2] = j * (1.0 / self.params.M) + (0.5 / self.params.M)
            data_table[k][3] = stimulate_matrix[i][j]

        return data_table

    def get_training_sets(self):
        if self.initialized is False:
            self.init_data()
        return self.train_list

    def get_testing_sets(self):
        if self.initialized is False:
            self.init_data()
        return self.test_list
