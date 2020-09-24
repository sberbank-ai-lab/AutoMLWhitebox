from copy import deepcopy
from typing import Dict, Union, List, Tuple, Hashable

import numpy as np
import pandas as pd

feature = Union[str, int, float]
f_list_type = List[feature]


class SmallNans:
    """
    Классс для обработки nan (вещественные признаки)
    для обработки маленьких групп и nan (категориальные признаки)
    Вещественные признаки в отдельную группу. Если сэмплов меньше, чем
    th_nan, то присавиваем woe 0. И на train и на test
    --------------------------------------------------------------------------------
    Категориальные признаки. Если категория небольшая (число сэмплов меньше, чем th_cat),
    то кодируем ее отельным числом. Если nan то кодируем по аналогии с
    вещественным случаем с помощью th_nan. Если на тесте встречаем категорию,
    которой не было на train, то отправляем ее в nan, маленькие категории, в woe со значением 0.
    """

    def __init__(self, th_nan: Union[int, float] = 32, th_cat: Union[int, float] = 32):
        """
        Parameters
        ----------
        th_nan
        th_cat
        """
        self._th_nan = th_nan
        self._th_cat = th_cat

        self.__features_type = None
        self.__cat_encoding = None  # Словарь с кодированием по группам категориальных признаков
        self.__all_encoding = None
        self.__spec_values = None

    def fit_transform(self, train: pd.DataFrame,
                      features_type: Dict[Hashable, str]) -> Tuple[pd.DataFrame, Dict[Hashable, Dict[str, float]]]:
        """

        Parameters
        ----------
        train
        features_type: Dict[Hashable, str]
           Типы признаков "cat" - категориальный, "real" - вещественный

        Returns
        -------

        """
        train_ = deepcopy(train)
        all_encoding = dict()
        cat_encoding = dict()
        spec_values = dict()
        self.__features_type = features_type
        for col in self.__features_type:
            d = dict()
            if self.__features_type[col] == "cat":
                vc = train_[col].value_counts()
                big_cat = set(vc.index)
                vc = vc.loc[vc < self._th_cat]
                vc_sum, small_cat = vc.sum(), set(vc.index)
                if vc_sum < self._th_nan:
                    enc_type = "__Small_0__"
                    d[enc_type] = 0
                    # Случай когда суммарно всех небольших категорий все равно мало
                else:
                    enc_type = "__Small__"
                    d[enc_type] = None
                train_.loc[train_[col].isin(small_cat), col] = enc_type
                cat_encoding[col] = big_cat.difference(small_cat), small_cat, enc_type
                #  Небольшие категории, которые будем кодировать отдельно
                
                
                
                
            nan_count = train_[col].isna().sum()
            if nan_count < self._th_nan:
                
                
                
                
                
                enc_type = "__NaN_0__"  # Число пропусков мало. WoE = 0
                d[enc_type] = 0
            else:
                enc_type = "__NaN__"  # Большое число пропусков. Кодируем как обычную категорию
                d[enc_type] = None
            spec_values[col] = d

            train_[col] = train_[col].fillna(enc_type)
            all_encoding[col] = enc_type

        self.__cat_encoding = cat_encoding
        self.__all_encoding = all_encoding
        self.__spec_values = spec_values

        return train_, spec_values

    def transform(self, test: pd.DataFrame, features: f_list_type, cat_merge_to: str = "to_small"):
        """

        Parameters
        ----------
        test:
            Тестовая выборка

        features:
            Список признаков для теста

        cat_merge_to: str
            В какую группу отпралять категории, которых не было в обучающей выборке
            "to_nan" -- в группу nan,
            "to_small" -- в категории маленьких групп,
            "to_woe_0" -- отдельная группа с woe == 0

        Returns
        -------

        """
        test_ = test[features].values.astype(object)
        spec_values = deepcopy(self.__spec_values.copy())
        for col in features:

            null_elements = pd.isnull(test_[:, features.index(col)])
            # и здесь фикс от Антона
            if self.__features_type[col] == "cat":
                big_cat, _, small_pad = self.__cat_encoding[col]
                if cat_merge_to == "to_woe_0":
                    fill_val = "__Small_0__"
                    spec_values[col][fill_val] = 0
                elif cat_merge_to == "to_small":
                    fill_val = small_pad
                elif cat_merge_to == "to_nan":
                    fill_val = self.__all_encoding[col]
                else:
                    raise ValueError("features_type is to_woe_0, to_nan or to_small")
                    
                
                test_[~np.isin(test_[:, features.index(col)], list(big_cat)), features.index(col)] = fill_val

            test_[null_elements, features.index(col)] = self.__all_encoding[col]

        return pd.DataFrame(test_, index=test.index, columns=features), spec_values

