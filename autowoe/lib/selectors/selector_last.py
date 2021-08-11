"""Post-selection."""

from typing import Dict
from typing import List
from typing import Tuple
from typing import TypeVar

import pandas as pd

from ..utilities.utils import Result
from ..utilities.utils import TaskType
from .composed_selector import ComposedSelector
from .l1 import L1
from .utils import F_LIST_TYPE


__all__ = ["Selector"]

WoE = TypeVar("WoE")


class Selector:
    """Class for post-selection of features.

    Args:
        interpreted_model: Build interpreted model.
        task: Task.
        train: Train features.
        target: Train target.
        features_type: Features types.
        n_jobs: Number of threads.
        cv_split: Cross-Val splits.

    """

    def __init__(
        self,
        interpreted_model: bool,
        task: TaskType,
        train: pd.DataFrame,
        target: pd.Series,
        features_type: Dict[str, str],
        n_jobs: int,
        cv_split: Dict[int, Tuple[List[int], List[int]]],
    ):

        self.__features_fit = list(features_type.keys())
        self.__pearson_selector = ComposedSelector(train, target, task)
        self.__main_selector = L1(
            task, train=train, target=target, interpreted_model=interpreted_model, n_jobs=n_jobs, cv_split=cv_split
        )
        self.train = train
        self.target = target

        self.__interpreted_model = interpreted_model
        self.__n_jobs = n_jobs
        self.__features = train.columns
        self.__cv_split = cv_split

    @property
    def features_fit(self):
        """Input features."""
        return self.__features_fit

    def __call__(
        self,
        feature_history: Dict[str, str],
        pearson_th: float,
        vif_th: float,
        metric_th: float,
        l1_grid_size: int,
        l1_exp_scale: float,
        metric_tol: float = 1e-4,
    ) -> Tuple[F_LIST_TYPE, Result]:
        """Run selector.

        Args:
            pearson_th: Pearson threshold.
            vif_th: VIF threshold
            metric_th: Metric threshold.
            l1_grid_size: Number of points on grid.
            l1_exp_scale: Maximum values of `C`.
            metric_tol: Metric tolerance.
            feature_history: HIstory of features filtering.

        Returns:
            Selected features, summary L1-selector info.

        """
        features_fit = self.__pearson_selector(
            feature_history,
            self.features_fit,
            pearson_th=pearson_th,
            metric_th=metric_th,
            vif_th=vif_th,
        )
        features_before = set(features_fit)
        features_fit, result = self.__main_selector(
            features_fit=features_fit, l1_grid_size=l1_grid_size, l1_exp_scale=l1_exp_scale, metric_tol=metric_tol
        )
        if feature_history is not None:
            features_diff = features_before - set(features_fit)
            for feat in features_diff:
                feature_history[feat] = f"Pruned by {self.__main_selector.__class__.__name__} selector"

        return features_fit, result
