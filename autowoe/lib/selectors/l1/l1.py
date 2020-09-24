import numpy as np
import pandas as pd

from copy import deepcopy
from typing import Union, Dict, List, Tuple, TypeVar

from ..utilities.utilities_sklearn import Result, l1_select

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

        Parameters
        ----------
        interpreted_model
        train
        target
        n_jobs
        cv_split
        """
        self.train = train
        self.target = target

        self.__interpreted_model = interpreted_model
        self.__n_jobs = n_jobs
        self.__features = train.columns
        self.__cv_split = cv_split

    @staticmethod
    def get_feature_sample(feature_weights: pd.Series) -> f_list_type:
        """

        Parameters
        ----------
        feature_weights: pd.Series

        Returns
        -------

        """
        while True:
            x_ = deepcopy(feature_weights)
            x_ = x_.map(lambda x: np.random.choice(a=2, size=1, p=[1 - x, x])[0])
            feature_sample = list(x_.loc[x_ == 1].index)
            if len(feature_sample) > 0:
                return feature_sample

    def __call__(self, features_fit: f_list_type,
                 l1_base_step: float,
                 l1_exp_step: float,
                 early_stopping_rounds: int, 
                 auc_tol: float = 1e-4) -> Tuple[f_list_type, Result]:
        """
        Точка входа в постотбор признаков

        Parameters
        ----------
        features_fit: f_list_type

        l1_base_step: float
            Базовый шаг уведичения коэффициента в l1 регулризации

        l1_exp_step: float
            Нелинейный коэффициент увеличения шага в l1 регуляризации

        early_stopping_rounds:

        Returns
        -------

        """
        np.random.seed(323)
        features_fit_ = features_fit.copy()
        dataset = self.train[features_fit_], self.target

        # TODO Перенести параметры из __init__ в __call__ ?
        best_features, result = l1_select(interpreted_model=self.__interpreted_model,
                                          n_jobs=self.__n_jobs,
                                          dataset=dataset,
                                          l1_base_step=l1_base_step,
                                          l1_exp_step=l1_exp_step,
                                          early_stopping_rounds=early_stopping_rounds,
                                          cv_split=self.__cv_split,
                                          verbose=True,
                                          auc_tol=auc_tol)

        return best_features, result
