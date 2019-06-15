import itertools as itertool

from PIL import Image
from pylab import *

import utils.math_utils as math_utils


class Params:
    M = 0
    N = M * M

    phi = 0.16
    sigma2 = 1

    neighbor_relative_ratio = 0.2

    k_fold_train_by_small = True
    k_fold_splits = 10

    def set_m(self, m):
        self.M = m
        self.N = m * m

    def get_params_string(self):
        param = 'M=' + self.M.__str__() + '\nphi = ' + self.phi.__str__() + '\nsigma2 = ' + self.sigma2.__str__() + \
                '\nneighbor_relative_ratio = ' + self.neighbor_relative_ratio.__str__()
        return param

    def get_stimulate_mu_cov(self):
        im = array(Image.open('resources/origin_data.jpg').convert('L'), 'f')
        M_0 = np.size(im, 0)
        sample_mu_matrix = np.zeros((self.M, self.M))
        sample_mu_list = np.zeros(self.N)
        sample_data_table = np.zeros((self.N, 4), dtype=float)
        ij = itertool.product(range(self.M), range(self.M))

        for k, (i, j) in enumerate(ij):
            sample_mu_matrix[i][j] = im[i * int(M_0 / self.M)][j * int(M_0 / self.M)]
            sample_data_table[k][0] = k
            sample_data_table[k][1] = i * (1.0 / self.M) + (0.5 / self.M)
            sample_data_table[k][2] = j * (1.0 / self.M) + (0.5 / self.M)
            sample_data_table[k][3] = sample_mu_matrix[i][j]
            sample_mu_list[k] = sample_mu_matrix[i][j]

        plt.matshow(sample_mu_matrix, cmap='Greys_r')
        plt.savefig("output/sample_mu_matrix.png")

        v_matrix = math_utils.get_v_matrix(self, sample_data_table[:, 1], sample_data_table[:, 2], self.N)

        plt.matshow(v_matrix, cmap='Greys_r')
        plt.savefig("output/v_matrix.png")

        return sample_mu_list, v_matrix

    # absolute exponential kernel
    def exponential_covariance_function(self, dist):
        exponential_variance = self.sigma2 * np.exp(-dist / self.phi)
        return exponential_variance

    def once_differentiable_matern_covariance_function(self, dist):
        matern_variance = self.sigma2 * (1 + (np.sqrt(3) * dist) / self.phi) * np.exp(-(np.sqrt(3) * dist) / self.phi)
        return matern_variance

    def twice_differentiable_matern_covariance_function(self, dist_0):
        poly_term = 1 + (np.sqrt(5) * dist_0 / self.phi) + ((5 * dist_0 * dist_0) / (3 * self.phi * self.phi))
        matern_variance = self.sigma2 * poly_term * np.exp(-(np.sqrt(5) * dist_0 / self.phi))
        return matern_variance
