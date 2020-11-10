import numpy as np
import pandas as pd
import lightgbm as lgb

from sklearn.model_selection import StratifiedKFold
from autowoe.lib.utilities.utils import flatten


class HTransform:
    """
    """

    def __init__(self, x: pd.Series, y: pd.Series, cv_splits: int = 5):
        """

        Args:
            x:
            y:
            cv_splits:
        """
        self.x, self.y = x, y
        self.cv = self._get_cv(cv_splits)

    @staticmethod
    def _get_cv(cv_splits: int) -> StratifiedKFold:
        """

        Args:
            cv_splits:

        Returns:

        """
        return StratifiedKFold(n_splits=cv_splits, random_state=323, shuffle=True)

    def __call__(self, tree_params: dict) -> np.ndarray:
        """
        Функция, возвращающая границы разбиения по переданной выборки и параметрам

        Args:
            tree_params: dict or lightgbm tree params

        Returns:

        """
        default_tree_params = {
            "boosting_type": "rf",
            "objective": "binary",
            "bagging_freq": 1,
            "bagging_fraction": 0.999,
            "feature_fraction": 0.999,
            "bagging_seed": 323,
            "verbosity": -1}

        unite_params = {**default_tree_params, **tree_params}
        lgb_train = lgb.Dataset(self.x.values.astype(np.float32)[:, np.newaxis], label=self.y)
        gbm = lgb.train(params=unite_params, train_set=lgb_train, num_boost_round=1)

        d_tree_prop = flatten(gbm.dump_model()["tree_info"][0])
        limits = {d_tree_prop[key] for key in d_tree_prop if "threshold" in key}

        limits = list(limits)
        limits.sort()

        return np.unique(limits)
