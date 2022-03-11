"""Process nan values."""

from collections import defaultdict
from copy import deepcopy
from typing import Any
from typing import Dict
from typing import Hashable
from typing import Optional
from typing import Set
from typing import Tuple
from typing import TypeVar
from typing import Union

import pandas as pd

from autowoe.lib.selectors.utils import F_LIST_TYPE


TKey = TypeVar("TKey")
TValue = TypeVar("TValue")


def _opt2val(name: str, options: Set[str]) -> Dict[str, str]:
    fmt = "__{NAME}_{VAL}__"
    return {k: fmt.format(NAME=name, VAL=k.rsplit("_")[-1]) for k in options}


def _values(d: Dict[TKey, TValue]) -> Set[TValue]:
    return {v for _, v in d.items()}


DEFAULT_OPTIONS_SPECIAL_VALUES: Set[str] = {"to_woe_0", "to_maxfreq", "to_minp", "to_maxp"}
EXTEND_OPTIONS_SPECIAL_VALUES: Set[str] = {*DEFAULT_OPTIONS_SPECIAL_VALUES, "to_nan"}

NAN_MERGE_CASES = _opt2val("NaN", DEFAULT_OPTIONS_SPECIAL_VALUES)
SMALL_MERGE_CASES = _opt2val("Small", EXTEND_OPTIONS_SPECIAL_VALUES)
MARK_MERGE_CASES = _opt2val("Mark", EXTEND_OPTIONS_SPECIAL_VALUES)


NAN_SET = {*_values(NAN_MERGE_CASES), "__NaN__"}
SMALL_SET = {*_values(SMALL_MERGE_CASES), "__Small__"}
MARK_SET = {*_values(MARK_MERGE_CASES), "__Mark__"}

CATEGORY_SPECIAL_SET = {*SMALL_SET, *NAN_SET, *MARK_SET} - {"__NaN__", "__Small__", "__Mark__"}
REAL_SPECIAL_SET = {*NAN_SET, *MARK_SET}  # - {"__NaN__", "__Small__", "__Mark__"}


def is_mark_prefix(s):
    """Mark encode."""
    return isinstance(s, str) and s.startswith("__Mark__")


class FeatureSpecialValues:
    """Class for processing special values in features.

    Вещественные признаки в отдельную группу. Если сэмплов меньше, чем
    th_nan, то присавиваем woe 0. И на train и на test
    --------------------------------------------------------------------------------
    Категориальные признаки. Если категория небольшая (число сэмплов меньше, чем th_cat),
    то кодируем ее отельным числом. Если nan то кодируем по аналогии с
    вещественным случаем с помощью th_nan. Если на тесте встречаем категорию,
    которой не было на train, то отправляем ее в nan, маленькие категории, в woe со значением 0.

    Groups of special values:
        1. NaN-values (real, categorical features).
        2. Small groups (categorical features).
        3. Mark values (real, categorical features).

    Real features processing:
        1. If there are fewer samples than `th_nan`, then assign `WoE` to 0.

    Categorical features processing:
        1. Small category (number of samples less than `th_cat`) ->
        2. Processing NaN-values as in real variables.
        3. Сategory that didn't occur in the train dataset is assigned a NaN.
        4. Сategory that didn't occur in the train dataset is assigned a NaN.

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
        th_mark: Union[int, float] = 32,
        cat_merge_to: str = "to_woe_0",
        nan_merge_to: str = "to_woe_0",
        mark_merge_to: str = "to_woe_0",
        mark_values: Optional[Dict[str, Any]] = None,
    ):
        self._th_nan = th_nan
        self._th_cat = th_cat
        self._th_mark = th_mark
        self._cat_merge_to = cat_merge_to
        self._nan_merge_to = nan_merge_to
        self._mark_merge_to = mark_merge_to
        self._mark_values = mark_values

        self._features_type = None
        self.cat_encoding = None  # Словарь с кодированием по группам категориальных признаков
        self.all_encoding = None
        self.mark_encoding = None
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
        mark_encoding = defaultdict(dict)
        spec_values = dict()
        self._features_type = features_type
        for col in self._features_type:
            d = dict()

            if self._mark_values is not None and col in self._mark_values:
                mark_values_mask = train_[col].isin(self._mark_values[col])

                fill_val = None
                if mark_values_mask.sum() < self._th_mark:
                    enc_type = MARK_MERGE_CASES[self._mark_merge_to]
                    if enc_type == "__Mark_0__":
                        fill_val = 0
                    # d[enc_type] = fill_val
                else:
                    enc_type = "__Mark__"

                    # if self._features_type[col] != "cat":
                    #     d[enc_type] = None

                for mv in self._mark_values[col]:
                    enc_type_t = enc_type + "{}__".format(mv) if enc_type == "__Mark__" else enc_type
                    train_.loc[train_[col] == mv, col] = enc_type_t
                    mark_encoding[col][mv] = enc_type_t
                    # if self._features_type[col] != "cat":
                    d[enc_type_t] = fill_val
            else:
                mark_values_mask = pd.Series(data=[False] * train_.shape[0], index=train_.index)

            if self._features_type[col] == "cat":
                vc = train_.loc[~mark_values_mask, col].value_counts()
                big_cat = set(vc.index)
                vc = vc.loc[vc < self._th_cat]
                vc_sum, small_cat = vc.sum(), set(vc.index)
                if vc_sum < self._th_nan:  # TODO: _th_nan -> _th_cat ?
                    # Случай когда суммарно всех небольших категорий все равно мало
                    enc_type = SMALL_MERGE_CASES[self._cat_merge_to]
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
        self.mark_encoding = mark_encoding
        self._spec_values = spec_values

        return train_, spec_values

    def transform(self, test: pd.DataFrame, features: F_LIST_TYPE):
        """Transform dataset.

        Args:
            test: Test dataset.
            features: List of features for processing.

        Returns:
            Processed dataset.

        """
        test_ = test[features].copy()

        for col in features:
            if self._mark_values is not None and col in self._mark_values:
                mark_values_mask = test_[col].isin(self._mark_values[col])
                if mark_values_mask.sum() > 0:
                    test_.loc[mark_values_mask, col] = test_.loc[mark_values_mask, col].map(self.mark_encoding[col])
            else:
                mark_values_mask = pd.Series(data=[False] * test.shape[0], index=test.index)

            if self._features_type[col] == "cat":
                big_cat, _, small_pad = self.cat_encoding[col]
                test_.loc[~(test_[col].isin(big_cat) | test_[col].isnull() | mark_values_mask), col] = small_pad

            test_[col] = test_[col].fillna(self.all_encoding[col])

        return test_, deepcopy(self._spec_values)
