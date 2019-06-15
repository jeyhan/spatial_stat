from datetime import datetime

from methods import approximation_method as approx, tradational_method as tradit
from utils import data_utils
from utils import log_utils
from utils import params_utils


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
        r2_approximation_method, cond_num_traditional_method = approx.approximation_method(training_set, testing_set,
                                                                                           params)

        avg_r2_approximation_method += r2_approximation_method
        avg_cond_num_approximation_method += cond_num_traditional_method

    avg_r2_approximation_method = avg_r2_approximation_method / params.k_fold_splits
    avg_cond_num_approximation_method = avg_cond_num_approximation_method / params.k_fold_splits

    return avg_r2_approximation_method, avg_cond_num_approximation_method


def main():
    for m in [10, 20]:
        params = params_utils.Params()
        params.set_m(m)
        log_utils.log('----------------------')
        log_utils.log('Params:')
        log_utils.log(params.get_params_string())

        data_set = data_utils.DataSet(params)

        log_utils.log('Traditional method running...')
        a = datetime.now()
        avg_r2_traditional_method, avg_cond_traditional_method = run_traditional_method(data_set, params)
        b = datetime.now()

        log_utils.log('Approximation method running...')
        avg_r2_approximation_method, avg_cond_approximation_method = run_approximation_method(data_set, params)
        c = datetime.now()

        log_utils.log('[Traditional] Avg R^2 value traditional is:{}'.format(avg_r2_traditional_method))
        log_utils.log('[Approximation] Avg R^2 value approximation is:{}'.format(avg_r2_approximation_method))

        log_utils.log('[Traditional] Average conditional number is: {}'.format(avg_cond_traditional_method.__str__()))
        log_utils.log(
            '[Approximation] Average conditional number is: {}'.format(avg_cond_approximation_method.__str__()))

        log_utils.log('\nTime cost traditional is:{} seconds'.format((b - a).seconds))
        log_utils.log('\nTime cost approximation is:{} seconds'.format((c - b).seconds))


main()
