from datetime import datetime

from methods import approximation_method as approx, tradational_method as tradit
from utils import data_utils
from utils import params_utils


def run_traditional_method():
    training_sets = data_utils.get_training_sets()
    testing_sets = data_utils.get_testing_sets()

    avg_r2_traditional_method = 0
    for i in range(0, params_utils.Params.k_fold_splits):
        training_set = training_sets[i]
        testing_set = testing_sets[i]
        r2_traditional_method = tradit.traditional_method(training_set, testing_set)
        avg_r2_traditional_method += r2_traditional_method
    avg_r2_traditional_method = avg_r2_traditional_method / params_utils.Params.k_fold_splits
    return avg_r2_traditional_method


def run_approximation_method():
    training_sets = data_utils.get_training_sets()
    testing_sets = data_utils.get_testing_sets()

    avg_r2_approximation_method = 0
    for i in range(0, params_utils.Params.k_fold_splits):
        training_set = training_sets[i]
        testing_set = testing_sets[i]
        r2_approximation_method = approx.approximation_method(training_set, testing_set)
        avg_r2_approximation_method += r2_approximation_method
    avg_r2_approximation_method = avg_r2_approximation_method / params_utils.Params.k_fold_splits
    return avg_r2_approximation_method


def main():
    data_utils.init_data()

    print('\nTraditional method.')
    a = datetime.now()
    avg_r2_traditional_method = run_traditional_method()
    b = datetime.now()

    print('\nApproximation method.')
    avg_r2_approximation_method = run_approximation_method()
    c = datetime.now()

    print('\nAvg R^2 value traditional is:{}'.format(avg_r2_traditional_method))
    print('\nAvg R^2 value approximation is:{}'.format(avg_r2_approximation_method))

    print('\nTime cost traditional is:{} seconds'.format((b - a).seconds))
    print('\nTime cost approximation is:{} seconds'.format((c - b).seconds))


# todo: use k-fold vaildation
main()
