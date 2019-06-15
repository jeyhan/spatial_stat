from datetime import datetime

import matplotlib.pyplot as plt

from methods import approximation_method as approx, tradational_method as tradit
from utils import data_utils
from utils import log_utils
from utils import params_utils


class Stimulation:
    experiment_type = 1  # 1: traditional, 2: approximation.
    v_matrix_dimension = 0
    avg_r2 = 0
    avg_cond = 0
    time_cost = 0


def run_traditional_method(data_set, params):
    training_sets = data_set.get_training_sets()
    testing_sets = data_set.get_testing_sets()

    avg_r2_traditional_method = 0
    avg_cond_num_traditional_method = 0

    for i in range(0, params.k_fold_splits):
        training_set = training_sets[i]
        testing_set = testing_sets[i]
        r2_traditional_method, cond_num_traditional_method = tradit.traditional_method(training_set, testing_set,
                                                                                       params)
        avg_r2_traditional_method += r2_traditional_method
        avg_cond_num_traditional_method += cond_num_traditional_method

    avg_r2_traditional_method = avg_r2_traditional_method / params.k_fold_splits
    avg_cond_num_traditional_method = avg_cond_num_traditional_method / params.k_fold_splits

    return avg_r2_traditional_method, avg_cond_num_traditional_method


def run_approximation_method(data_set, params):
    training_sets = data_set.get_training_sets()
    testing_sets = data_set.get_testing_sets()

    avg_r2_approximation_method = 0
    avg_cond_num_approximation_method = 0
    for i in range(0, params.k_fold_splits):
        training_set = training_sets[i]
        testing_set = testing_sets[i]
        r2_approximation_method, cond_traditional_method = approx.approximation_method(training_set, testing_set,
                                                                                       params)

        avg_r2_approximation_method += r2_approximation_method
        avg_cond_num_approximation_method += cond_traditional_method

    avg_r2_approximation_method = avg_r2_approximation_method / params.k_fold_splits
    avg_cond_num_approximation_method = avg_cond_num_approximation_method / params.k_fold_splits

    return avg_r2_approximation_method, avg_cond_num_approximation_method


def main():
    dims = [20, 30, 40, 50, 60, 80, 100]

    simulations = []
    for m in dims:
        params = params_utils.Params()
        params.set_m(m)
        log_utils.log('----------------------')
        log_utils.log('Params:')
        log_utils.log(params.get_params_string())

        data_set = data_utils.DataSet(params)

        log_utils.log('Core running...')
        a = datetime.now()
        avg_r2_traditional_method, avg_cond_traditional_method = run_traditional_method(data_set, params)
        b = datetime.now()

        avg_r2_approximation_method, avg_cond_approximation_method = run_approximation_method(data_set, params)
        c = datetime.now()

        stimulation_trad = Stimulation()
        stimulation_trad.experiment_type = 1
        stimulation_trad.v_matrix_dimension = params.N
        stimulation_trad.avg_r2 = avg_r2_traditional_method
        stimulation_trad.avg_cond = avg_cond_traditional_method
        stimulation_trad.time_cost = (b - a).seconds
        simulations.append(stimulation_trad)

        stimulation_appro = Stimulation()
        stimulation_appro.experiment_type = 2
        stimulation_appro.v_matrix_dimension = int(params.neighbor_relative_ratio * params.N)
        stimulation_appro.avg_r2 = avg_r2_approximation_method
        stimulation_appro.avg_cond = avg_cond_approximation_method
        stimulation_appro.time_cost = (c - b).seconds
        simulations.append(stimulation_appro)

        log_utils.log('[Traditional] V matrix dim = {}'.format(params.N))
        log_utils.log('[Traditional] Avg R^2 value traditional is:{}'.format(avg_r2_traditional_method))
        log_utils.log('[Traditional] Average conditional number is: {}'.format(avg_cond_traditional_method.__str__()))
        log_utils.log('[Traditional] Time cost is:{} seconds'.format((b - a).seconds))

        log_utils.log('[Approximation] V matrix dim = {}'.format(int(params.neighbor_relative_ratio * params.N)))
        log_utils.log('[Approximation] Avg R^2 value approximation is:{}'.format(avg_r2_approximation_method))
        log_utils.log(
            '[Approximation] Average conditional number is: {}'.format(avg_cond_approximation_method.__str__()))
        log_utils.log('[Approximation] Time cost is:{} seconds'.format((c - b).seconds))

    avg_cond_traditional_method_list = []
    avg_cond_approximation_method_list = []

    for i in range(0, dims.__len__()):
        avg_cond_traditional_method_list.append(simulations[2 * i].avg_cond)
        avg_cond_approximation_method_list.append(simulations[2 * i + 1].avg_cond)

    plt.figure(figsize=(8, 4))  # 创建绘图对象
    plt.title('Result Analysis')
    plt.plot(dims, avg_cond_traditional_method_list, color='green', label='Traditional Avg Cond')
    plt.plot(dims, avg_cond_approximation_method_list, color='red', label='Approximation Avg Cond')
    plt.legend()  # 显示图例

    plt.xlabel('M')
    plt.ylabel('Conditional Number')
    plt.savefig("output/dim_to_cond.png")


main()
