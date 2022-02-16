"""Selector based on Lasso."""

from typing import Dict
from typing import List
from typing import Tuple
from typing import TypeVar

import numpy as np
import pandas as pd

from autowoe.lib.selectors.utils import Result
from autowoe.lib.selectors.utils import l1_select
from autowoe.lib.utilities.utils import TaskType

from .utils import F_LIST_TYPE
from .utils import FEATURE


WoE = TypeVar("WoE")


class L1:
    """L1 selector.

    Args:
        interpreted_model: Build interpreted model.
        train: Train features.
        target: Train target.
        n_jobs: Number of threads.
        cv_split: Cross-Val splits.

    """

    def __init__(
        self,
        task: TaskType,
        interpreted_model: bool,
        train: pd.DataFrame,
        target: pd.Series,
        n_jobs: int,
        cv_split: Dict[int, Tuple[List[int], List[int]]],
    ):
        self.task = task
        self.train = train
        self.target = target

        self.__interpreted_model = interpreted_model
        self.__n_jobs = n_jobs
        self.__features = train.columns
        self.__cv_split = cv_split

    def __call__(
        self, features_fit: List[FEATURE], l1_grid_size: int, l1_exp_scale: float, metric_tol: float = 1e-4
    ) -> Tuple[F_LIST_TYPE, Result]:
        """Run selector.

        Args:
            features_fit: List of features.
            l1_grid_size: Number of points on grid.
            l1_exp_scale: Maximum value of `C`.
            metric_tol: Metric tolerance.

        Returns:
            Selected features, summary info.


        """
        np.random.seed(323)
        features_fit_ = features_fit.copy()
        dataset = self.train[features_fit_], self.target

        best_features, result = l1_select(
            self.task,
            interpreted_model=self.__interpreted_model,
            n_jobs=self.__n_jobs,
            dataset=dataset,
            l1_grid_size=l1_grid_size,
            l1_exp_scale=l1_exp_scale,
            cv_split=self.__cv_split,
            metric_tol=metric_tol,
        )

        return best_features, result
