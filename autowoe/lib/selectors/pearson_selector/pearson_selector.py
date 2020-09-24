import pandas as pd

from typing import Dict, Tuple, Union, List, TypeVar
from collections import OrderedDict
from itertools import combinations

import numpy as np

# from scipy.stats import pearsonr

WoE = TypeVar("WoE")
feature = Union[str, int, float]
f_list_type = List[feature]


class PearsonSelector:
    """
    Класс для отбора признаков по корреляции Пирсона
    """

    def __init__(self, train: pd.DataFrame,
                 target: pd.Series,
                 features_type: Dict[str, str],
                 woe_dict: Dict[Union[str, int, float], WoE]):
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
        self.iv = self.__get_iv(features_type, woe_dict)

    @staticmethod
    def __get_iv(features_type: Dict[str, str], woe_dict: Dict[feature, WoE]) -> Dict[feature, float]:
        """
        Parameters
        ----------
        features_type
        woe_dict

        Returns
        -------

        """
        iv = dict()
        for feature in features_type:
            iv[feature] = woe_dict[feature].iv
        return iv

#     @staticmethod
#     def __corr_in_intervals(corr: float, pearson_th: List[Tuple[float, float]]) -> bool:
#         for interval in pearson_th:
#             if interval[0] < corr < interval[1]:
#                 return True
#         return False

    def __call__(self, features_fit: f_list_type, pearson_th: float) -> f_list_type:
        """

        Parameters
        ----------
        features_fit
        pearson_th
           Трешхолд по которому отсеиваются высокоскоррелированные признаки

        Returns
        -------

        """
        features_fit_ = features_fit.copy()
        if len(features_fit_) <= 1:
            return features_fit_
        
        # Опять чиним за Пенкиным)
        df = self.train[features_fit_]
        # предрассчитаем корреляции + удалим значения с 1 бином
        corrs = df.corr().abs()
        cols = np.array(corrs.columns)
        sing_vals = np.array([corrs.loc[x, x] for x in cols])
        # print(sing_vals)
        sing_vals = np.array(corrs.columns)[np.isnan(sing_vals)]
         
        features_not_fit = []
        
#         # ivs
#         ivs = np.array([self.iv[x] for x in cols])
#         order_ = ivs.argsort()[::-1]
        
#         # начинаем с высоких IV
#         sorted_cols = cols[order_]
#         for n, col in enumerate(sorted_cols[:-1]):
#             candidates = sorted_cols[n+1:]
#             scores_ = corrs.loc[col, candidates]
#             to_drop = list(scores_.index[scores_ > pearson_th].values)
#             features_not_fit.extend(to_drop)
        
        iv = OrderedDict({x for x in self.iv.items() if x[0] in features_fit_})
        iv = OrderedDict(sorted(iv.items(), key=lambda x: -x[1]))
        all_comb = list(combinations(iv.keys(), 2))
        for pair in enumerate(all_comb):  # Перебор всех пар в порядке убывания iv
            if not set(pair[1]).intersection(features_not_fit):
                corr = corrs.loc[pair[1][0], pair[1][1]]
                # corr, _ = pearsonr(df[pair[1][0]].values, df[pair[1][1]].values)
                if corr > pearson_th:
                    features_not_fit.append(pair[1][1])  # если по корреляции не проходит добавляем с меньшим iv
        return list(set(features_fit_) - set(features_not_fit) - set(sing_vals)) 
