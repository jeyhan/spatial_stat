import itertools as itertool

from PIL import Image
from pylab import *

import utils.math_utils as math_utils


class Params:
    M = 60
    N = M * M

    phi = 0.16
    sigma2 = 1

    neighbor_relative_ratio = 0.2

    k_fold_train_by_small = True
    k_fold_splits = 10


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
