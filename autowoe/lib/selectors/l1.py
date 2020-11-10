from typing import Union, Dict, List, Tuple, TypeVar

import numpy as np
import pandas as pd

from autowoe.lib.selectors.utils import Result, l1_select

WoE = TypeVar("WoE")
feature = Union[str, int, float]
f_list_type = List[feature]


class L1:
    """
    Класс для постотбора признаков
    Простая раеадизация. Алгоритм с закручиванием регуляризации
    """

    def __init__(self, interpreted_model: bool, train: pd.DataFrame, target: pd.Series, n_jobs: int,
                 cv_split: Dict[int, Tuple[List[int], List[int]]]):
        """

        Args:
            interpreted_model:
            train:
            target:
            n_jobs:
            cv_split:
        """
        self.train = train
        self.target = target

        self.__interpreted_model = interpreted_model
        self.__n_jobs = n_jobs
        self.__features = train.columns
        self.__cv_split = cv_split

    def __call__(self, features_fit: f_list_type,
                 l1_grid_size: int,
                 l1_exp_scale: float,
                 auc_tol: float = 1e-4) -> Tuple[f_list_type, Result]:
        """
        Точка входа в постотбор признаков

        Args:
            features_fit: f_list_type
            l1_grid_size: int
                Базовый шаг уведичения коэффициента в l1 регулризации
            l1_exp_scale: float
                Нелинейный коэффициент увеличения шага в l1 регуляризации
            auc_tol:

        Returns:

        """
        np.random.seed(323)
        features_fit_ = features_fit.copy()
        dataset = self.train[features_fit_], self.target

        best_features, result = l1_select(interpreted_model=self.__interpreted_model,
                                          n_jobs=self.__n_jobs,
                                          dataset=dataset,
                                          l1_grid_size=l1_grid_size,
                                          l1_exp_scale=l1_exp_scale,
                                          cv_split=self.__cv_split,
                                          auc_tol=auc_tol)

        return best_features, result
