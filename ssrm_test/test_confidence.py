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
from functools import reduce

import numpy as np
from scipy.stats import multinomial

from . import constants as const
from .confidence import confidence_sequence, confidence_sequence_from_posteriors
from .ssrm_test import sequential_posteriors


def test_confidence_sequence_from_posteriors():
    """
    Tests that the confidence sequence is a running intersection
    """
    theta = np.array([0.5, 0.5])
    dirichlet_probability = np.array([0.5, 0.5])
    dirichlet_concentration = 1
    sample_size = 400
    observations = multinomial.rvs(1, theta, size=sample_size)
    posteriors = sequential_posteriors(
        observations,
        theta,
        dirichlet_probability=dirichlet_probability,
        dirichlet_concentration=dirichlet_concentration,
    )
    cis = confidence_sequence_from_posteriors(posteriors, 0.99)
    lowers = [ci[0] for ci in cis]
    uppers = [ci[1] for ci in cis]
    assert np.all(np.diff(uppers) <= 0)  # upper bound non increasing
    assert np.all(np.diff(lowers) >= 0)  # lower bound non decreasing


def test_confidence():
    """
    Tests that the confidence sequence is a running intersection
    """
    theta = np.array([0.5, 0.5])
    sample_size = 4000
    observations = multinomial.rvs(1, theta, size=sample_size)
    cis = confidence_sequence(observations, 0.99)
    lowers = [ci[0] for ci in cis]
    uppers = [ci[1] for ci in cis]
    assert np.all(np.diff(uppers) <= 0)  # upper bound non increasing
    assert np.all(np.diff(lowers) >= 0)  # lower bound non decreasing
