"""Contains a data generating process used in the thesis, based on IHDP covariates.

The helpers begin with an underscore, as they are not exported and should not be
used by themselves. `multi_expo_on_ihdp` makes the entry point to access this DGP.

Basic Usage: ::

    >>> replications = multi_expo_on_ihdp(setting="multi-modal", n_replications=100)

"""
from typing import List, Optional, Union

import numpy as np
from numpy.random import RandomState
from scipy.special import expit
from sklearn.utils import check_random_state

from ..sets.ihdp import get_ihdp_covariates
from ..utils import Frame, generate_data

OptRandState = Optional[Union[int, RandomState]]


def _multi_modal_effect(covariates, random_state):
    prob = expit(covariates[:, 0]) > 0.5
    return random_state.normal((3 * prob) + 1 * (1 - prob), 0.1)


def _exponential_effect(covariates):
    return np.exp(1 + expit(covariates[:, 0]))


def _polynomial_effect(covariates, random_state):
    return random_state.normal(2*covariates["x_8"]*covariates["x_7"] + covariates["x_8"]*covariates["x_9"] + 3*covariates["x_7"]*covariates["x_9"], 0, size = len(covariates))


def _linear_effect(covariates, random_state):
    return np.sum(covariates[:])


def _sinusoidal_effect(covariates, random_state):
    return np.sin(covariates[:, 0])


def _multi_outcome(covariates, *, random_state: RandomState, **kwargs):
    random_state = check_random_state(random_state)
    y_0 = random_state.normal(0, 0.2, size=len(covariates))
    y_1 = y_0 + _multi_modal_effect(covariates, random_state)
    mu_0, mu_1 = y_0, y_1
    return mu_0, mu_1, y_0, y_1


def _expo_outcome(covariates, *, random_state: RandomState, **kwargs):
    random_state = check_random_state(random_state)
    y_0 = random_state.normal(0, 0.2, size=len(covariates))
    y_1 = y_0 + _exponential_effect(covariates)
    mu_0, mu_1 = y_0, y_1
    return mu_0, mu_1, y_0, y_1


def _polynomial_outcome(covariates, *, random_state: RandomState, **kwargs):
    random_state = check_random_state(random_state)
    y_0 = random_state.normal(0, 0.2, size=len(covariates))
    y_1 = y_0 + _polynomial_effect(covariates)
    mu_0, mu_1 = y_0, y_1
    return mu_0, mu_1, y_0, y_1


def _linear_outcome(covariates, *, random_state: RandomState, **kwargs):
    random_state = check_random_state(random_state)
    y_0 = random_state.normal(0, 0.2, size=len(covariates))
    y_1 = y_0 + _linear_effect(covariates, random_state)
    mu_0, mu_1 = y_0, y_1
    return mu_0, mu_1, y_0, y_1


def _sinusoidal_outcome(covariates, *, random_state: RandomState, **kwargs):
    random_state = check_random_state(random_state)
    y_0 = random_state.normal(0, 0.2, size=len(covariates))
    y_1 = y_0 + _sinusoidal_effect(covariates, random_state)
    mu_0, mu_1 = y_0, y_1
    return mu_0, mu_1, y_0, y_1


def _treatment_assignment(covariates, *, random_state: RandomState, **kwargs):
    random_state = check_random_state(random_state)
    return random_state.binomial(1, p=expit(covariates[:, 0]))


def _rct_treatment_assignment(covariates, *, random_state: RandomState, **kwargs):
    random_state = check_random_state(random_state)
    return random_state.binomial(1, 0.5)


def _single_confounder_treatment_assignment(covariates, *, random_state: RandomState, **kwargs):
    random_state = check_random_state(random_state)
    return random_state.binomial(1, p=expit(covariates[:, 0]))


def _multi_confounder_treatment_assignment(covariates, *, random_state: RandomState, **kwargs):
    random_state = check_random_state(random_state)
    return random_state.binomial(1, p=expit(covariates[:, 0]))


def dgp_on_ihdp(
    setting: str = "multi-modal",
    treatment_assignment_setting = "rct",
    n_samples: int = None,
    n_replications: int = 1,
    random_state: OptRandState = None,
) -> List[Frame]:

    covariates = get_ihdp_covariates().values

    if setting == "multi-modal":
        outcome = _multi_outcome
    elif setting == "linear":
        outcome = _linear_outcome
    elif setting == "sinusoidal":
        outcome = _sinusoidal_outcome
    elif setting == "polynomial":
        outcome = _polynomial_outcome
    else:
        outcome = _expo_outcome
    
    if treatment_assignment_setting == "rct":
        treatment_assignment = _rct_treatment_assignment
    elif treatment_assignment_setting == "single_confounder":
        treatment_assignment = _single_confounder_treatment_assignment
    elif treatment_assignment_setting == "multi_confounder":
        treatment_assignment = _multi_confounder_treatment_assignment
    else:
        treatment_assignment = _treatment_assignment

    return generate_data(
        covariates,
        treatment_assignment,
        outcome,
        n_samples=n_samples,
        n_replications=n_replications,
        random_state=random_state,
    )


def multi_expo_on_ihdp(
    setting: str = "multi-modal",
    n_samples: int = None,
    n_replications: int = 1,
    random_state: OptRandState = None,
) -> List[Frame]:
    r"""Reproduces a specific DGP used in the Thesis based on IHDP covariates.

    The DGP has been designed to show that some learners,
    specifically tree-based learner, are very good in a multi-modal
    settings, while in a smooth exponential setting, linear learners are much better
    suited to estimate individual treatment effects.
    For details, see Chapter 5.3.1 (p.56) in the Thesis.

    .. math ::

        y_0 &= N(0, 0.2) \\
        y_1 &= y_0 + tau \\
        t &= \text{BERN}[sigmoid(X_8)]

    and tau is either exponential

    .. math :: \tau = exp(1 + sigmoid(X_8))

    or multi-modal

    .. math ::

        c &= \mathbb{I}(sigmoid(X_8) > 0.5) \\
        \tau &= N(3*c + (1 - c), 0.1)


    Args:
        setting: either 'multi-modal' or 'exponential'
        n_samples: number of instances
        n_replications: number of replications
        random_state: a random state from which to draw

    Returns:
        a list of CausalFrames for each replication requested

    """
    # Use covariates as nd.array
    covariates = get_ihdp_covariates().values

    if setting == "multi-modal":
        outcome = _multi_outcome
    else:
        outcome = _expo_outcome

    return generate_data(
        covariates,
        _treatment_assignment,
        outcome,
        n_samples=n_samples,
        n_replications=n_replications,
        random_state=random_state,
    )
