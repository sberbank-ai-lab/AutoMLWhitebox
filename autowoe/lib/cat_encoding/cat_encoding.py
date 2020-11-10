from copy import deepcopy
from typing import Dict, List, Union

import numpy as np
import pandas as pd


class CatEncoding:
    """
    Class for categorical data converting/reconverting to float values
    """

    def __init__(self, data: pd.DataFrame):
        """

        Args:
            data: Данные для кодирования. I колонка - признак, II - таргет
        """
        self.data = data
        self.col = data.columns

        self.data_info = pd.DataFrame(index=data.index, columns=[self.col[0], "mean_enc"])
        self.data_info[self.col[0]] = self.data[self.col[0]].values

    def __call__(self, cv_index_split: Dict[int, List[int]], nan_index: np.array,
                 cat_alpha: float = 1.) -> pd.DataFrame:
        """
        Mean_target encoding на кросс-валидации

        Args:
            cv_index_split:
            nan_index:
            cat_alpha:

        Returns:

        """
        cv_index_split_ = deepcopy(cv_index_split)
        feature, target = self.col

        for key in cv_index_split_:
            train_index, test_index = cv_index_split_[key]
            train_index, test_index = np.setdiff1d(train_index, nan_index), np.setdiff1d(test_index, nan_index)

            data_sl = self.data.iloc[train_index]
            d_agg = data_sl.groupby(feature)[target].agg(['sum', 'count'])
            d_agg = (d_agg['sum'] + cat_alpha * data_sl[target].mean()) / (d_agg['count'] + cat_alpha)

            d_agg = d_agg.to_dict()
            self.data_info.iloc[test_index, 1] = self.data_info.iloc[test_index, 0].map(d_agg)

        train_f = self.data.copy()
        train_f.iloc[:, 0] = self.data_info["mean_enc"].values
        return train_f

    def mean_target_reverse(self, split: Union[List[float], np.ndarray]) -> Dict[int, int]:
        """
        Обратная к операция к mean_target_encoding.
        По заданному split определяет категории.
        Должна быть запущена только после __call__

        Args:
            split:

        Returns:

        """
        df = self.data_info.copy()
        df["split_cat"] = np.searchsorted(split, df.mean_enc.values)

        crosstab = pd.crosstab(df[self.col[0]], df.split_cat)
        crosstab = crosstab.div(crosstab.sum(axis=1), axis=0)
        max_cat = np.argmax(crosstab.values, axis=1)

        # словарь соответствий: имя категории -> номер бина
        return dict(zip(crosstab.index, max_cat))
