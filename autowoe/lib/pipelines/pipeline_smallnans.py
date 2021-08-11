"""Process nan values."""

from copy import deepcopy
from typing import Dict
from typing import Hashable
from typing import List
from typing import Tuple
from typing import Union

import pandas as pd


feature = Union[str, int, float]
f_list_type = List[feature]

CAT_MERGE_CASES = {
    "to_woe_0": "__Small_0__",
    "to_maxfreq": "__Small_maxfreq__",
    "to_minp": "__Small_minp__",
    "to_maxp": "__Small_maxp__",
    "to_nan": "Small_nan__",
}

NAN_MERGE_CASES = {
    "to_woe_0": "__NaN_0__",
    "to_maxfreq": "__NaN_maxfreq__",
    "to_minp": "__NaN_minp__",
    "to_maxp": "__NaN_maxp__",
}


class SmallNans:
    """Классс для обработки nan (вещественные признаки) для обработки маленьких групп и nan (категориальные признаки).

    Вещественные признаки в отдельную группу. Если сэмплов меньше, чем
    th_nan, то присавиваем woe 0. И на train и на test
    --------------------------------------------------------------------------------
    Категориальные признаки. Если категория небольшая (число сэмплов меньше, чем th_cat),
    то кодируем ее отельным числом. Если nan то кодируем по аналогии с
    вещественным случаем с помощью th_nan. Если на тесте встречаем категорию,
    которой не было на train, то отправляем ее в nan, маленькие категории, в woe со значением 0.

    Args:
        th_nan: Threshold for NaN-values process.
        th_cat: Threshold for category values process.
        cat_merge_to:
        nan_merge_to:

    """

    def __init__(
        self,
        th_nan: Union[int, float] = 32,
        th_cat: Union[int, float] = 32,
        cat_merge_to: str = "to_woe_0",
        nan_merge_to: str = "to_woe_0",
    ):
        self._th_nan = th_nan
        self._th_cat = th_cat
        self._cat_merge_to = cat_merge_to
        self._nan_merge_to = nan_merge_to

        self._features_type = None
        self.cat_encoding = None  # Словарь с кодированием по группам категориальных признаков
        self.all_encoding = None
        self._spec_values = None

    def fit_transform(
        self, train: pd.DataFrame, features_type: Dict[Hashable, str]
    ) -> Tuple[pd.DataFrame, Dict[Hashable, Dict[str, float]]]:
        """Fit/transform.

        Args:
            train: Dataset.
            features_type: Type of features. {"cat" - category, "real" - real}

        Returns:
            Processed dataset, special values.

        """
        train_ = deepcopy(train)
        all_encoding = dict()
        cat_encoding = dict()
        spec_values = dict()
        self._features_type = features_type
        for col in self._features_type:
            d = dict()

            if self._features_type[col] == "cat":
                vc = train_[col].value_counts()
                big_cat = set(vc.index)
                vc = vc.loc[vc < self._th_cat]
                vc_sum, small_cat = vc.sum(), set(vc.index)
                if vc_sum < self._th_nan:
                    # Случай когда суммарно всех небольших категорий все равно мало
                    enc_type = CAT_MERGE_CASES[self._cat_merge_to]
                    fill_val = 0 if enc_type == "__Small_0__" else None
                    d[enc_type] = fill_val
                else:
                    enc_type = "__Small__"
                    # d[enc_type] = None

                train_.loc[train_[col].isin(small_cat), col] = enc_type
                cat_encoding[col] = big_cat.difference(small_cat), small_cat, enc_type
                #  Небольшие категории, которые будем кодировать отдельно

            nan_count = train_[col].isna().sum()

            if nan_count < self._th_nan:
                enc_type = NAN_MERGE_CASES[self._nan_merge_to]
                fill_val = 0 if enc_type == "__NaN_0__" else None
                d[enc_type] = fill_val
            else:
                enc_type = "__NaN__"  # Большое число пропусков. Кодируем как обычную категорию
                # исключаем NaN из специальных значений для категорий
                if self._features_type[col] != "cat":
                    d[enc_type] = None

            spec_values[col] = d

            train_[col] = train_[col].fillna(enc_type)
            all_encoding[col] = enc_type

        self.cat_encoding = cat_encoding
        self.all_encoding = all_encoding
        self._spec_values = spec_values

        return train_, spec_values

    def transform(self, test: pd.DataFrame, features: f_list_type):
        """Transform dataset.

        Args:
            test: Test dataset.
            features: List of features for processing.

        Returns:
            Processed dataset.

        """
        test_ = test[features].copy()

        for col in features:
            if self._features_type[col] == "cat":
                big_cat, _, small_pad = self.cat_encoding[col]
                test_.loc[~(test_[col].isin(big_cat) | test_[col].isnull()), col] = small_pad

            test_[col] = test_[col].fillna(self.all_encoding[col])

        return test_, deepcopy(self._spec_values)
