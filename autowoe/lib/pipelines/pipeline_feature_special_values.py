"""Process nan values."""

from copy import deepcopy
from typing import Any
from typing import Dict
from typing import Final
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


DEFAULT_OPTIONS_SPECIAL_VALUES: Final[Set[str]] = {"to_woe_0", "to_maxfreq", "to_minp", "to_maxp"}
EXTEND_OPTIONS_SPECIAL_VALUES: Final[Set[str]] = {*DEFAULT_OPTIONS_SPECIAL_VALUES, "to_nan"}

NAN_MERGE_CASES: Final = _opt2val("NaN", DEFAULT_OPTIONS_SPECIAL_VALUES)
SMALL_MERGE_CASES: Final = _opt2val("Small", EXTEND_OPTIONS_SPECIAL_VALUES)
MARKED_MERGE_CASES: Final = _opt2val("Marked", EXTEND_OPTIONS_SPECIAL_VALUES)


NAN_SET: Final = {*_values(NAN_MERGE_CASES), "__NaN__"}
SMALL_SET: Final = {*_values(SMALL_MERGE_CASES), "__Small__"}
MARKED_SET: Final = {*_values(MARKED_MERGE_CASES), "__Marked__"}

CATEGORY_SPECIAL_SET: Final = {*SMALL_SET, *NAN_SET, *MARKED_SET} - {"__NaN__", "__Small__", "__Marked__"}
REAL_SPECIAL_SET: Final = {*NAN_SET, *MARKED_SET}  # - {"__NaN__", "__Small__", "__Marked__"}


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
        3. Marked values (real, categorical features).

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
        marked_values: Optional[Dict[str, Any]] = None,
    ):
        self._th_nan = th_nan
        self._th_cat = th_cat
        self._th_mark = th_mark
        self._cat_merge_to = cat_merge_to
        self._nan_merge_to = nan_merge_to
        self._mark_merge_to = mark_merge_to
        self._marked_values = marked_values

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
        mark_encoding = dict()
        spec_values = dict()
        self._features_type = features_type
        for col in self._features_type:
            d = dict()

            if self._marked_values is not None and col in self._marked_values:
                marked_values_mask = train_[col].isin(self._marked_values[col])

                if marked_values_mask.sum() < self._th_mark:
                    enc_type = MARKED_MERGE_CASES[self._mark_merge_to]
                    fill_val = 0 if enc_type == "__Marked_0__" else None
                    d[enc_type] = fill_val
                else:
                    enc_type = "__Marked__"

                    if self._features_type[col] != "cat":
                        d[enc_type] = None

                train_.loc[marked_values_mask, col] = enc_type
                mark_encoding[col] = enc_type
            else:
                marked_values_mask = pd.Series(data=[False] * train_.shape[0], index=train_.index)

            if self._features_type[col] == "cat":
                vc = train_.loc[~marked_values_mask, col].value_counts()
                big_cat = set(vc.index)
                vc = vc.loc[vc < self._th_cat]
                vc_sum, small_cat = vc.sum(), set(vc.index)
                if 0 < vc_sum < self._th_nan:  # TODO: _th_nan -> _th_cat ?
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
            if self._features_type[col] == "cat":
                big_cat, _, small_pad = self.cat_encoding[col]
                test_.loc[~(test_[col].isin(big_cat) | test_[col].isnull()), col] = small_pad

            test_[col] = test_[col].fillna(self.all_encoding[col])

            if self._marked_values is not None and col in self._marked_values:
                marked_values_mask = test_[col].isin(self._marked_values[col])
                if marked_values_mask.sum() > 0:
                    test_.loc[marked_values_mask, col] = self.mark_encoding[col]

        return test_, deepcopy(self._spec_values)
