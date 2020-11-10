import pandas as pd

from typing import Union, Dict, List, Tuple, TypeVar

from .composed_selector import ComposedSelector
from .l1 import L1

from ..utilities.utils import Result

__all__ = ["Selector"]

WoE = TypeVar("WoE")
feature = Union[str, int, float]
f_list_type = List[feature]


class Selector:
    """
    Класс для постотбора признаков
    """

    def __init__(self, interpreted_model: bool, train: pd.DataFrame, target: pd.Series, features_type: Dict[str, str],
                 n_jobs: int, cv_split: Dict[int, Tuple[List[int], List[int]]]):
        """

        Args:
            interpreted_model:
            train:
            target:
            features_type:
            n_jobs:
            cv_split:
        """
        self.__features_fit = list(features_type.keys())
        self.__pearson_selector = ComposedSelector(train, target)
        self.__main_selector = L1(train=train,
                                  target=target,
                                  interpreted_model=interpreted_model,
                                  n_jobs=n_jobs,
                                  cv_split=cv_split)
        self.train = train
        self.target = target

        self.__interpreted_model = interpreted_model
        self.__n_jobs = n_jobs
        self.__features = train.columns
        self.__cv_split = cv_split

    @property
    def features_fit(self):
        """

        Returns:

        """
        return self.__features_fit

    def __call__(self, pearson_th: float,
                 vif_th: float,
                 auc_th: float,
                 l1_grid_size: int,
                 l1_exp_scale: float,
                 auc_tol: float = 1e-4,
                 feature_history: Dict[str, str] = None) -> Tuple[f_list_type, Result]:
        """

        Args:
            pearson_th:
            vif_th:
            auc_th:
            l1_grid_size:
            l1_exp_scale:
            auc_tol:
            feature_history:

        Returns:

        """
        features_fit = self.__pearson_selector(self.features_fit, pearson_th=pearson_th, auc_th=auc_th, vif_th=vif_th,
                                               feature_history=feature_history)
        features_before = set(features_fit)
        features_fit, result = self.__main_selector(features_fit=features_fit,
                                                    l1_grid_size=l1_grid_size,
                                                    l1_exp_scale=l1_exp_scale,
                                                    auc_tol=auc_tol)
        if feature_history is not None:
            features_diff = features_before - set(features_fit)
            for feat in features_diff:
                feature_history[feat] = f'Pruned by {self.__main_selector.__class__.__name__} selector'

        return features_fit, result
