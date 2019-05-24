from methods import approximation_method as approx, tradational_method as tradit
from utils import params_utils
from utils import data_utils
from datetime import datetime


def main():
    data_utils.init_data()

    print('\nTraditional method.')
    a = datetime.now()

    r2_traditional_method = tradit.traditional_method()
    b = datetime.now()

    print('\nApproximation method.')
    r2_approximation_method = approx.approximation_method()
    c = datetime.now()

    print("\nModel split_ratio={}".format(params_utils.Params.training_testing_split_ratio))

    print('\nR^2 value traditional is:{}'.format(r2_traditional_method))
    print('\nR^2 value approximation is:{}'.format(r2_approximation_method))

    print('\nTime cost traditional is:{} seconds'.format((b - a).seconds))
    print('\nTime cost approximation is:{} seconds'.format((c - b).seconds))


main()
