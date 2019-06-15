import itertools as itertool

from PIL import Image
from pylab import *

import utils.math_utils as math_utils


class Params:
    M = 20
    N = M * M

    phi = 0.16
    sigma2 = 1

    neighbor_relative_ratio = 0.2

    k_fold_train_by_small = True
    k_fold_splits = 10


# absolute exponential kernel
def exponential_covariance_function(dist):
    exponential_variance = Params.sigma2 * np.exp(-dist / Params.phi)
    return exponential_variance


def once_differentiable_matern_covariance_function(dist):
    matern_variance = Params.sigma2 * (1 + (np.sqrt(3) * dist) / Params.phi) * np.exp(-(np.sqrt(3) * dist) / Params.phi)
    return matern_variance


def twice_differentiable_matern_covariance_function(dist_0):
    poly_term = 1 + (np.sqrt(5) * dist_0 / Params.phi) + ((5 * dist_0 * dist_0) / (3 * Params.phi * Params.phi))
    matern_variance = Params.sigma2 * poly_term * np.exp(-(np.sqrt(5) * dist_0 / Params.phi))
    return matern_variance


def get_stimulate_mu_cov():
    im = array(Image.open('resources/origin_data.jpg').convert('L'), 'f')
    M_0 = np.size(im, 0)
    sample_mu_matrix = np.zeros((Params.M, Params.M))
    sample_mu_list = np.zeros(Params.N)
    sample_data_table = np.zeros((Params.N, 4), dtype=float)
    ij = itertool.product(range(Params.M), range(Params.M))

    for k, (i, j) in enumerate(ij):
        sample_mu_matrix[i][j] = im[i * int(M_0 / Params.M)][j * int(M_0 / Params.M)]
        sample_data_table[k][0] = k
        sample_data_table[k][1] = i * (1.0 / Params.M) + (0.5 / Params.M)
        sample_data_table[k][2] = j * (1.0 / Params.M) + (0.5 / Params.M)
        sample_data_table[k][3] = sample_mu_matrix[i][j]
        sample_mu_list[k] = sample_mu_matrix[i][j]

    plt.matshow(sample_mu_matrix, cmap='Greys_r')
    plt.savefig("output/sample_mu_matrix.png")

    v_matrix = math_utils.get_v_matrix(sample_data_table[:, 1], sample_data_table[:, 2], Params.N)

    plt.matshow(v_matrix, cmap='Greys_r')
    plt.savefig("output/v_matrix.png")

    return sample_mu_list, v_matrix
