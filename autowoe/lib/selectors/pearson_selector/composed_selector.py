import pandas as pd


from sklearn.metrics import roc_auc_score
from typing import Dict, Tuple, Union, List, TypeVar
from collections import OrderedDict
from itertools import combinations

import numpy as np
import scipy as sp
from copy import copy

WoE = TypeVar("WoE")
feature = Union[str, int, float]
f_list_type = List[feature]


class ComposedSelector:
    """
    Класс для отбора признаков по критериям:
    1) одно уникальное woe
    2) auc одномерный меньше порога
    3) VIF признака выше порога
    4) существуют признаки с парной корреляцией выше порога
    """

    def __init__(self, train: pd.DataFrame,
                 target: pd.Series):
        """

        Parameters
        ----------
        train
        target
        features_type
        woe_dict
        """
        self.train = train
        self.target = target
        # precompute corrs 
        cc = np.abs(sp.corrcoef(train.values, rowvar=False))
        self.precomp_corr = pd.DataFrame(cc, index=train.columns, columns=train.columns)
        self.precomp_aucs = pd.Series([1 - roc_auc_score(target, train[x]) for x in train.columns],
                                     index=train.columns)
        
    @staticmethod
    def __compare_msg(closure, value, msg=None):
        flg = closure(value)
        if not flg:
            print(msg)
        return flg
    
    def __call__(self, features_fit: f_list_type, pearson_th: float = .9, 
                 auc_th: float = .5, vif_th: float = 5., feature_history: Dict[str, str] = None) -> f_list_type:
        
        candidates = copy(features_fit)
        features_before = set(candidates)
        # откинем константные 
        candidates = [col for col in candidates if self.__compare_msg(
                                    lambda x: ~np.isnan(self.precomp_corr.loc[x, x]), col, 
                                    'Feature {0} removed due to single WOE value'.format(col))
                     ]
        features_after = set(candidates)
        features_diff = features_before - features_after
        if feature_history is not None:
            for feature in features_diff:
                feature_history[feature] = 'Constant WOE value'
        # откинем с низким ауком
        features_before = features_after
        candidates = [col for col in candidates if self.__compare_msg(
                                    lambda x: self.precomp_aucs[x] >= auc_th, col, 
                                    'Feature {0} removed due to low AUC value {1}'.format(col, self.precomp_aucs[col]))
                     ]
        features_after = set(candidates)
        features_diff = features_before - features_after
        if feature_history is not None:
            for feature in features_diff:
                feature_history[feature] = f'Low AUC value = {round(self.precomp_aucs[feature], 2)}'
        
        # итеративный виф
        max_vif = np.inf
        while max_vif > vif_th:
            corrs = self.precomp_corr.loc[candidates, candidates]
            # fix singularity
            corrs = corrs.values + np.diag(np.ones(corrs.shape[0]) * 1e-4)
            VIFs = np.linalg.inv(corrs).diagonal()
            
            max_vif_idx = VIFs.argmax()
            max_vif = VIFs[max_vif_idx]

            if max_vif >= vif_th:
                print('Feature {0} removed due to high VIF value = {1}'.format(candidates[max_vif_idx], max_vif))
                if feature_history is not None:
                    feature_history[candidates[max_vif_idx]] = f'High VIF value = {round(max_vif, 2)}'
                candidates = [x for (n, x) in enumerate(candidates) if n != max_vif_idx]               
        
        # попарные корреляции
        # отсортируем по убыванию аука
        order_ = np.array([self.precomp_aucs[x] for x in candidates]).argsort()[::-1]
        candidates = [candidates[x] for x in order_]
        
        n = 0
        while n < (len(candidates) - 1):
            
            partial_corrs = self.precomp_corr.loc[candidates[n], candidates[n + 1:]]
            big_partial_corrs = partial_corrs[partial_corrs >= pearson_th]
            if len(big_partial_corrs) > 0:
                print('Features {0}: aucs = {1} was removed due to corr = {2} with feature {3}: auc = {4}'.format(
                        list(big_partial_corrs.index.values),
                        list(self.precomp_aucs[big_partial_corrs.index]),
                        list(big_partial_corrs.values),
                        candidates[n],
                        self.precomp_aucs[candidates[n]]
                        ))
                if feature_history is not None:
                    for feature in big_partial_corrs.index.values:
                        feature_history[feature] = f'High correlation with feature {candidates[n]}'
            
            candidates = [x for x in candidates if x not in set(big_partial_corrs.index.values)]
            n += 1
            
        return candidates