import numpy as np
from scipy.optimize import bisect
from scipy.special import gammaln, loggamma, xlogy
from ssrm_test.ssrm_test import accumulator
from toolz import curry
from toolz.itertoolz import accumulate

from . import constants as const


def confidence_sequence(posteriors, coverage):

    assert (
        len(posteriors[0][const.DATA]) == 2
    ), "Confidence Sequences only implemented for Binomial data"
    initial_ci = [0, 1]

    @curry
    def tighten_ci(coverage, ci, posterior):
        l, u = ci
        succ, fail = posterior[const.DATA]
        log_bayes_factor = posterior[const.LOG_BAYES_FACTOR]
        p_0 = posterior[const.POSTERIOR_M0][0]
        x = succ
        n = succ + fail

        def is_inside_ci(p):
            """
            Returns a positive number if p is inside the confidence interval, else negative
            """
            threshold = (
                log_bayes_factor
                + x * np.log(p_0)
                + (n - x) * np.log(1 - p_0)
                + np.log(1 - coverage)
            )
            return xlogy(x, p) + xlogy(n - x, 1 - p) - threshold

        mle = x / n
        if is_inside_ci(u) > 0:
            upper = u
        else:
            assert mle < u, "MLE is greater than or equal to current upper bound"
            upper = bisect(is_inside_ci, mle, u)
        assert upper <= u, "upper bound increased"

        if is_inside_ci(l) > 0:
            lower = l
        else:
            assert mle > l, "MLE is less than or equal to current lower bound"
            lower = bisect(is_inside_ci, l, mle)
        assert lower >= l, "lower bound decreased"
        confidence = [lower, upper]
        return confidence

    return list(accumulate(tighten_ci(coverage), posteriors, initial_ci))
