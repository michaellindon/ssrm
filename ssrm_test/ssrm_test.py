#     Copyright 2020 Optimizely Inc.
#
#     Licensed under the Apache License, Version 2.0 (the "License");
#     you may not use this file except in compliance with the License.
#     You may obtain a copy of the License at
#
#         http://www.apache.org/licenses/LICENSE-2.0
#
#     Unless required by applicable law or agreed to in writing, software
#     distributed under the License is distributed on an "AS IS" BASIS,
#     WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#     See the License for the specific language governing permissions and
#     limitations under the License.

import warnings
from typing import List

import numpy as np
from scipy.special import gammaln, loggamma, xlogy
from toolz.itertoolz import accumulate

from . import constants as const


def update_posterior(n: np.ndarray, alpha: np.ndarray) -> np.ndarray:
    """
    Updates posterior distribution.

    Parameters
    ----------
    n : np.ndarray
        Amount to update posterior distribution parameter by.
    alpha : np.ndarray
        Posterior distribution parameter.

    Returns
    -------
    np.ndarray
        Updated alpha parameter, which defines the posterior.
    """
    return n + alpha


def log_posterior_predictive(n: np.ndarray, alpha: np.ndarray) -> float:
    """
    Log posterior predictive density resulting from marginalizing a
    multinomial likelihood with a dirichlet(alpha) prior, evaluated at n.

    Parameters
    ----------
    n : np.ndarray
        A new data point.
    alpha : np.ndarray
        Parameter defining the dirichlet distribution.

    Returns
    -------
    float
        log posterior predictive density evaluated at n.
    """

    return (
        loggamma(n.sum() + 1)
        - loggamma(n + 1).sum()
        + loggamma(alpha.sum())
        - loggamma(alpha).sum()
        + loggamma(alpha + n).sum()
        - loggamma((alpha + n).sum())
    )


def accumulator(acc: dict, new_data_point: np.ndarray) -> dict:
    """
    Binary operator for accumulating statistical quantities across a stream of data.

    Parameters
    ----------
    acc : dict
        Contains posterior parameters and log marginal likelihoods at previous data point.
    new_data_point : np.ndarray
        Number of counts in each variation.

    Returns
    -------
    dict
        A dictionary storing accumulated values with the same keys as in acc.
    """
    log_density_M1 = log_posterior_predictive(new_data_point, acc[const.POSTERIOR_M1])
    log_density_M0 = multinomiallogpmf(
        new_data_point, new_data_point.sum(), acc[const.POSTERIOR_M0]
    )
    log_marginal_likelihood_M1 = acc[const.LOG_MARGINAL_LIKELIHOOD_M1] + log_density_M1
    log_marginal_likelihood_M0 = acc[const.LOG_MARGINAL_LIKELIHOOD_M0] + log_density_M0
    log_bayes_factor = log_marginal_likelihood_M1 - log_marginal_likelihood_M0

    log_posterior_odds = acc[const.LOG_POSTERIOR_ODDS] + log_density_M1 - log_density_M0
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        bayes_factor = np.exp(log_bayes_factor)
        posterior_odds = np.exp(log_posterior_odds)
    post_prob = posterior_probability(posterior_odds)
    p_value = min(1 / bayes_factor, acc[const.P_VALUE])
    data = acc[const.DATA] + new_data_point
    out = {
        const.LOG_POSTERIOR_ODDS: log_posterior_odds,
        const.POSTERIOR_ODDS: posterior_odds,
        const.LOG_BAYES_FACTOR: log_bayes_factor,
        const.BAYES_FACTOR: bayes_factor,
        const.P_VALUE: p_value,
        const.POSTERIOR_PROBABILITY: post_prob,
        const.LOG_MARGINAL_LIKELIHOOD_M1: log_marginal_likelihood_M1,
        const.LOG_MARGINAL_LIKELIHOOD_M0: log_marginal_likelihood_M0
        if new_data_point.sum() > 0
        else acc[const.LOG_MARGINAL_LIKELIHOOD_M0],
        const.POSTERIOR_M1: acc[const.POSTERIOR_M1] + new_data_point,
        const.POSTERIOR_M0: acc[const.POSTERIOR_M0],
        const.DATA: data,
    }
    return out


@np.vectorize
def posterior_probability(posterior_odds: float) -> float:
    """
    Computes the posterior probability given posterior odds.

    Parameters
    ----------
    posterior_odds : float
        Posterior odds of an SRM.

    Returns
    -------
    float
        The posterior probability of an SRM.
    """
    if posterior_odds == np.Inf:
        return 1
    elif posterior_odds == -np.Inf:
        return 0
    else:
        return posterior_odds / (1.0 + posterior_odds)


def bayes_factor(posterior: dict) -> float:
    """
    Returns the Bayes Factor (BF) of an SRM.
    The BF can be np.Inf (numerically), which occurs when the data is
    overwhelmingly in favour of an SRM, but that is not an issue and is
    handled correctly by other functions.

    Parameters
    ----------
    posterior : dict
        Posterior distribution parameters.

    Returns
    -------
    float
        Bayes factor of an SRM.
    """
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        bf = np.exp(
            posterior[const.LOG_MARGINAL_LIKELIHOOD_M1]
            - posterior[const.LOG_MARGINAL_LIKELIHOOD_M0]
        )
    return bf


def sequential_bayes_factors(
    data: np.ndarray,
    null_probabilities: np.ndarray,
    dirichlet_probability=None,
    dirichlet_concentration=10000,
) -> np.ndarray:
    """
    Accumulates sequential Bayes factors after every datapoint.

    Parameters
    ----------
    data : np.ndarray
        Data i.e. [[0, 1], [1, 0], ...] or [[10, 12], [14, 3],...]
    null_probabilities : np.ndarray
        The expected traffic allocation probability, where the values must sum to 1.
        Note the order must match the order of data.
    dirichlet_probability : np.ndarray
        The mean of the dirichlet distribution, subject to elements summing to 1.
    dirichlet_concentration: float
        How tightly the prior concentrates around the mean.

    Returns
    -------
    np.ndarray
        Sequential Bayes factors after every datapoint.
    """
    data = np.array(data)
    if not _validate_data(data):
        raise TypeError("Data is supposed to be an array of integer arrays")
    posteriors = sequential_posteriors(
        data, null_probabilities, dirichlet_probability, dirichlet_concentration
    )
    return [posterior[const.BAYES_FACTOR] for posterior in posteriors]


def sequential_posterior_probabilities(
    data: np.ndarray,
    null_probabilities: np.ndarray,
    dirichlet_probability=None,
    dirichlet_concentration=10000,
    prior_odds=1,
) -> np.ndarray:
    """
    Accumulates sequential posterior probabilities after every datapoint.

    Parameters
    ----------
    data : np.ndarray
        Data i.e. [[0, 1], [1, 0], ...] or [[10, 12], [14, 3],...]
    null_probabilities : np.ndarray
        The expected traffic allocation probability, where the values must sum to 1.
        Note the order must match the order of data.
    dirichlet_probability : np.ndarray
        The mean of the dirichlet distribution, subject to elements summing to 1.
    dirichlet_concentration: float
        How tightly the prior concentrates around the mean.
    prior_odds: float
        Prior odds in favour of an SRM.

    Returns
    -------
    np.ndarray
        Sequential posterior probabilities after every datapoint.
    """
    data = np.array(data)
    if not _validate_data(data):
        raise TypeError("Data is supposed to be an array of integer arrays")
    posteriors = sequential_posteriors(
        data,
        null_probabilities,
        dirichlet_probability,
        dirichlet_concentration,
        prior_odds,
    )
    return [posterior[const.POSTERIOR_PROBABILITY] for posterior in posteriors]


def sequential_p_values(
    data: np.ndarray,
    null_probabilities: np.ndarray,
    dirichlet_probability=None,
    dirichlet_concentration=10000,
) -> np.ndarray:
    """
    Accumulates sequential p-values after every datapoint

    Parameters
    ----------
    data : np.ndarray
        Data i.e. [[0, 1], [1, 0], ...] or [[10, 12], [14, 3], ...]
    null_probabilities : np.ndarray
        The expected traffic allocation probability, where the values must sum to 1.
        Note the order must match the order of data.
    dirichlet_probability : np.ndarray
        The mean of the dirichlet distribution, subject to elements summing to 1.
    dirichlet_concentration : float
        How tightly the prior concentrates around the mean.

    Returns
    -------
    np.ndarray
        Sequential p-value after every datapoint.
    """
    data = np.array(data)
    if not _validate_data(data):
        raise TypeError("Data is supposed to be an array of integer arrays")
    posteriors = sequential_posteriors(
        data, null_probabilities, dirichlet_probability, dirichlet_concentration
    )
    return [posterior[const.P_VALUE] for posterior in posteriors]


def sequential_posteriors(
    data: np.ndarray,
    null_probabilities: np.ndarray,
    dirichlet_probability=None,
    dirichlet_concentration=10000,
    prior_odds=1,
) -> List[dict]:
    """
    Accumulates the posteriors and marginal likelihoods for each datapoint.

    Parameters
    ----------
    data : List[tuple]
        Data.
    null_probabilities : np.ndarray
        The expected traffic allocation probability, where the values must sum to 1.
        Note the order must match the order of data.
    dirichlet_probability : np.ndarray
        The mean of the dirichlet distribution, subject to elements summing to 1.
    dirichlet_concentration : float
        How tightly the prior concentrates around the mean.

    Returns
    -------
    List[dict]
        Posterior distribution and marginal likelihoods at every datapoint.

    Examples
    --------
    For unit-level data, we can use sequential_posteriors like so:
    >>> data = np.array([[1, 0, 0], [0, 1, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]])
    >>> null_probabilities = [0.4, 0.4, 0.2]
    >>> list_dict = sequential_posteriors(data, null_probabilities)

    For time-aggregated data, we can pass in tuples that represent the
    aggregated count during that time:
    >>> data = np.array([[20, 17, 9], [18, 21, 8], [4, 6, 4], [18, 19, 11]])
    >>> null_probabilities = [0.4, 0.4, 0.2]
    >>> list_dict = sequential_posteriors(data, null_probabilities)
    """
    data = np.array(data)
    if not _validate_data(data):
        raise TypeError("Data is supposed to be an array of integer arrays")
    null_probabilities = np.array(null_probabilities)
    if dirichlet_probability is None:
        dirichlet_probability = null_probabilities
    dirichlet_alpha = dirichlet_probability * dirichlet_concentration
    acc = {
        const.LOG_POSTERIOR_ODDS: np.log(prior_odds),
        const.POSTERIOR_ODDS: prior_odds,
        const.LOG_BAYES_FACTOR: 0,
        const.BAYES_FACTOR: 1,
        const.P_VALUE: 1,
        const.POSTERIOR_PROBABILITY: posterior_probability(prior_odds),
        const.LOG_MARGINAL_LIKELIHOOD_M1: 0,
        const.LOG_MARGINAL_LIKELIHOOD_M0: 0,
        const.POSTERIOR_M1: dirichlet_alpha,
        const.POSTERIOR_M0: null_probabilities,
        const.DATA: np.array([0] * len(null_probabilities)),
    }
    return list(accumulate(accumulator, data, acc))[1:]


def srm_test(data: np.ndarray, null_probabilities: np.ndarray) -> float:
    """
    Creates a Sample Ratio Mismatch (SRM) test to validate whether an
    experiment follows a predefined distribution of data amongst its
    variations.

    Parameters
    ----------
    data : np.ndarray
        Data.
    null_probabilities : np.ndarray
        The expected traffic allocation probability, where the values must sum to 1.
        Note the order must match the order of data.

    Returns
    -------
    float
        Final posterior probability of a sample ratio mismatch.

    Examples
    --------
    For unit-level data, we can use sequential_posteriors like so:
    >>> data = np.array([[1, 0, 0], [0, 1, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]])
    >>> null_probabilities = [0.4, 0.4, 0.2]
    >>> prob = srm_test(data, null_probabilities)

    For time-aggregated data, we can pass in tuples that represent the
    aggregated count during that time:
    >>> data = np.array([[20, 17, 9], [18, 21, 8], [4, 6, 4], [18, 19, 11]])
    >>> null_probabilities = [0.4, 0.4, 0.2]
    >>> prob = srm_test(data, null_probabilities)
    """
    data = np.array(data)
    final_posterior = sequential_posteriors(data, null_probabilities)[-1]
    final_bf = bayes_factor(final_posterior)
    post_prob = posterior_probability(final_bf)
    return post_prob


def get_bayes_factor_threshold(
    y: List[int], dirichlet_alpha: np.ndarray, alpha: float
) -> float:
    """
    Computes a constant which defines a constraint in the convex
    optimization program for finding confidence intervals over
    individual parameters.

    Parameters
    ----------
    y : List[int]
        Cumulative counts of visitors assigned to each arm.
    dirichlet_alpha : np.ndarray
        Parameter of dirichlet distribution.
    alpha : float
        Frequentist type 1 error probability.

    Returns
    -------
    float
        Value which defines the feasible set i.e. returns the c in c <= sum y_i log p_i.
    """
    return (
        log_posterior_predictive(y, np.array(dirichlet_alpha))
        + np.log(alpha)
        - loggamma(y.sum() + 1)
        + loggamma(y + 1).sum()
    )


def multinomiallogpmf(x: List[int], n: int, p: List[float]):
    """
    Alternative implementation of multinomial.logpmf from scipy.stats.
    This is about 5x faster.

    Parameters
    ----------
    x : List[int]
        A list of observations e.g. [10,21,33].
    n : int
        The number of trials (equal to sum(x)).
    p : List[float]
        A list of probabilities which should sum to 1 e.g. [0.2, 0.4, 0.4].

    Returns
    -------
    float
        Probability mass function of a multinomial(n,p) pmf evaluated at x.
    """
    x = np.array(x)
    p = np.array(p)
    return gammaln(n + 1) + np.sum(xlogy(x, p) - gammaln(x + 1), axis=-1)


@np.vectorize
def _is_integer(x):
    if isinstance(x, int):
        return True
    elif isinstance(x, float):
        return x.is_integer()
    else:
        return False


def _validate_data(data: np.array) -> bool:
    return np.all(_is_integer(data.flatten()))
