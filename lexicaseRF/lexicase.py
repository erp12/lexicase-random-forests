from copy import copy
from random import choice
import numpy as np
import numpy.random as npr


def median_absolute_deviation(a):
    """Returns the MAD of X.
    Parameters
    ----------
    a : array-like, shape = (n,)
    Returns
    -------
    mad : float
    """
    return np.median(np.abs(a - np.median(a)))


def epsilon_lexicase_selection(population):
    candidates = copy(population)
    cases = np.arange(len(population[0]._error_vector))
    npr.shuffle(cases)

    all_errors = np.array([i._error_vector for i in candidates])
    epsilon = np.apply_along_axis(median_absolute_deviation, 0,
                                  all_errors)

    while len(cases) > 0 and len(candidates) > 1:
        case = cases[0]
        errors_this_case = [i._error_vector[case] for i in candidates]
        best_val_for_case = min(errors_this_case)
        max_error = best_val_for_case + epsilon[case]

        def test(i): return i._error_vector[case] <= max_error
        candidates = [i for i in candidates if test(i)]
        cases = cases[1:]
    return choice(candidates)
