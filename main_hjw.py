from datetime import datetime

from methods import approximation_method as approx, tradational_method as tradit
from utils import data_utils
from utils import params_utils


def main():
    data_utils.init_data()
    training_set = data_utils.get_training_set()
    testing_set = data_utils.get_testing_set()

    print('\nTraditional method.')
    a = datetime.now()

    r2_traditional_method = tradit.traditional_method(training_set, testing_set)
    b = datetime.now()

    print('\nApproximation method.')
    r2_approximation_method = approx.approximation_method(training_set, testing_set)
    c = datetime.now()

    print("\nModel split_ratio={}".format(params_utils.Params.training_testing_split_ratio))

    print('\nR^2 value traditional is:{}'.format(r2_traditional_method))
    print('\nR^2 value approximation is:{}'.format(r2_approximation_method))

    print('\nTime cost traditional is:{} seconds'.format((b - a).seconds))
    print('\nTime cost approximation is:{} seconds'.format((c - b).seconds))


# todo: use k-fold vaildation
main()
