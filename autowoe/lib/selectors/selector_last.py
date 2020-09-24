import pandas as pd
import numpy as np

from typing import Union, Dict, List, Tuple, TypeVar, Optional

# from .pearson_selector.pearson_selector import PearsonSelector
from .pearson_selector.composed_selector import ComposedSelector
from .l1.l1 import L1
from .gen_add_del_l1.gen_add_del_l1 import GenAddDelL1

from ..utilities.utilities import Result

__all__ = ["Selector"]

WoE = TypeVar("WoE")
feature = Union[str, int, float]
f_list_type = List[feature]


class Selector:
    """
    Класс для постотбора признаков
    """

    def __init__(self, interpreted_model: bool, train: pd.DataFrame, target: pd.Series, features_type: Dict[str, str],
                 woe_dict: Dict[feature, WoE], n_jobs: int, cv_split: Dict[int, Tuple[List[int], List[int]]],
                 feature_groups_count: int, population_size: Optional[int], imp_type: str,
                 group_kf: Optional[np.ndarray] = None):
        """
        Parameters
        ----------
        interpreted_model
        train
        target
        features_type
        woe_dict
        n_jobs
        cv_split
        population_size
        group_kf
        """
        self.__features_fit = list(features_type.keys())
        # self.__pearson_selector = PearsonSelector(train, target, features_type, woe_dict)
        # для нового пирсон селектора
        self.__pearson_selector = ComposedSelector(train, target)
        self.__main_selector = (GenAddDelL1(train=train,
                                            target=target,
                                            interpreted_model=interpreted_model,
                                            n_jobs=n_jobs,
                                            cv_split=cv_split,
                                            feature_groups_count=feature_groups_count,
                                            population_size=population_size,
                                            imp_type=imp_type,
                                            group_kf=group_kf) if population_size else
                                L1(train=train,
                                   target=target,
                                   interpreted_model=interpreted_model,
                                   n_jobs=n_jobs,
                                   cv_split=cv_split))
        self.train = train
        self.target = target

        self.__interpreted_model = interpreted_model
        self.__n_jobs = n_jobs
        self.__features = train.columns
        self.__cv_split = cv_split

    @property
    def features_fit(self):
        """
        Returns
        -------

        """
        return self.__features_fit

    def __call__(self, pearson_th: float,
                 vif_th: float,
                 auc_th: float,
                 l1_base_step: float, l1_exp_step: float,
                 early_stopping_rounds: int, 
                 auc_tol: float = 1e-4,
                 feature_history: Dict[str, str] = None) -> Tuple[f_list_type, Result]:
        """
        Parameters
        ----------
        pearson_th
        l1_base_step
        l1_exp_step
        early_stopping_rounds

        Returns
        -------

        """
        features_fit = self.__pearson_selector(self.features_fit, pearson_th=pearson_th, auc_th=auc_th, vif_th=vif_th, feature_history=feature_history)
        features_before = set(features_fit)
        features_fit, result = self.__main_selector(features_fit=features_fit,
                                                    l1_base_step=l1_base_step,
                                                    l1_exp_step=l1_exp_step,
                                                    early_stopping_rounds=early_stopping_rounds,
                                                    auc_tol=auc_tol)
        if feature_history is not None:
            features_diff = features_before - set(features_fit)
            for feature in features_diff:
                feature_history[feature] = f'Pruned by {self.__main_selector.__class__.__name__} selector'
        
        return features_fit, result
