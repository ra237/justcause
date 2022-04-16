import numpy as np
from numpy.random import RandomState
from sklearn.utils import check_random_state
from sklearn.ensemble import RandomForestRegressor

from ...utils import int_from_random_state


class RandomForest:

    def __init__(
        self, num_trees: int = 200, random_state: RandomState = None,
        max_features="sqrt", n_jobs=-1, **kwargs
    ):
        self.random_state = check_random_state(random_state)

        self.forest = RandomForestRegressor(n_estimators=num_trees,
        random_state=self.random_state, max_features=max_features ,
        n_jobs=n_jobs, **kwargs)
        
        self.num_trees = num_trees
        self.kwargs = kwargs

    def __repr__(self):
        return self.__str__()

    def __str__(self):
        """Simple string representation for logs and outputs"""
        return ("{}(num_trees={})").format(
            self.__class__.__name__, self.num_trees
        )

    def estimate_ate(self, x: np.array, t: np.array, y: np.array) -> float:
        """ Estimates ATE of the given population
        Fits the CausalForest and predicts the ITE. The mean of all ITEs is
        returned as the ATE.
        Args:
            x: covariates
            t: treatment indicator
            y: factual outcome
        Returns: average treatment effect of the population
        """
        self.fit(x, t, y)
        ite = self.predict_ite(x)
        return float(np.mean(ite))

    def predict_ite(self, x: np.array, t=None, y=None) -> np.array:
        """
        Predicts ITE vor given samples without using facutals
        Args:
            x:
        Returns:
        """
        return self.forest.predict(x).flatten()

    def fit(self, x: np.array, t: np.array, y: np.array) -> None:
        """ Fits the forest using factual data"""

        self.forest.fit(x, y)