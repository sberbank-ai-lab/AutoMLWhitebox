"""Optimization of decision tree parameteres."""

from collections import OrderedDict
from copy import copy
from itertools import product
from typing import Any
from typing import Dict
from typing import Iterable
from typing import List
from typing import Tuple
from typing import Union

import lightgbm as lgb
import numpy as np
import pandas as pd

from autowoe.lib.utilities.cv_split_f import cv_split_f
from autowoe.lib.utilities.utils import TaskType


# TODO: Do we need random state here?
np.random.seed(232)


class TreeParamOptimizer:
    """Optimizer of decision tree parameters.

    Args:
        data: Dataset. First column - feature, second - Target.
        params_range: OrderedDict with parameters and ranges for binning algorithms
            Ex. params_range = OrderedDict({"max_depth": (4, 7, 17, 2, 3),  "min_child_samples": (40000, 20000, 5000),})

    """

    _cv_metric_map = {"auc": "auc", "mse": "l2"}

    def __init__(self, data: pd.DataFrame, task: TaskType, params_range: Dict[str, tuple], n_folds: int = 5):
        self._params_range = copy(params_range)
        self._task = task
        self._metric = "auc" if self._task == TaskType.BIN else "mse"

        ds_params = {}
        try:
            ds_params["min_data_in_bin"] = self._params_range.pop("min_data_in_bin")[0]
        except KeyError:
            pass

        # TODO: Fix double saved data
        self._X = pd.DataFrame(data.iloc[:, 0])
        self._y = data.iloc[:, 1]

        self._lgb_train = lgb.Dataset(data=self._X.copy(), label=self._y.copy(), params=ds_params)
        self.n_folds = n_folds
        self._params_stats = None

    def __get_folds(self, random_state):
        skf = cv_split_f(self._X, self._y, self._task, None, self.n_folds, random_state)

        # folds = np.zeros(self._lgb_train.data.shape[0])
        # for fold_idx, tt_idx in skf.items():
        #     _, test_idx = tt_idx
        #     folds[test_idx] =  fold_idx

        # return skf.items()

        for _, v in skf.items():
            yield v

    @property
    def __params_gen(self) -> Iterable[Tuple]:
        return product(*self._params_range.values())

    def __get_scores(self, params: Dict[str, Any], n: int) -> List[float]:
        """Scores for set of parameters.

        Args:
            params: Tree parameters.
            n: The amount of cross-validation to evaluate hyperparameters

        Returns:
            Scores.

        """
        default_tree_params = {
            "boosting_type": "gbdt",
            "learning_rate": 1,
            "objective": "binary" if self._task == TaskType.BIN else "regression",
            "bagging_freq": 1,
            "bagging_fraction": 1,
            "feature_fraction": 1,
            "bagging_seed": 323,
            "n_jobs": 1,
            "verbosity": -1,
        }
        unite_params = {**params, **default_tree_params}

        scores = []
        for seed in range(n):
            folds = self.__get_folds(seed)
            cv_results = lgb.cv(
                params=unite_params, train_set=self._lgb_train, num_boost_round=1, folds=folds, metrics=self._metric
            )
            scores.append(cv_results["{}-mean".format(self._cv_metric_map[self._metric])])

        return scores

    def __get_stats(self, stats: List[List[float]]):
        """Calculate statistics of scores.

        Args:
            stats: Scores [combinations of parameters, cv-s, number of folds in cv]

        """
        stats = np.array(stats)
        median_, std_ = np.median(stats, axis=(1, 2)), np.std(stats, axis=(1, 2))

        id_max = zip(*(median_, -std_))  # номер комбинации с наилучшим качеством
        id_max = max(enumerate(id_max), key=lambda x: x[1])[0]

        stat_score = zip(*(median_, std_))
        self._params_stats = OrderedDict((key, value) for (key, value) in zip(self.__params_gen, stat_score)), id_max

    def __call__(self, n: int) -> Dict[str, Union[int, str, None]]:
        """Execute optimization.

        Args:
            n: Number of iterations.

        Returns:
            Best parameters.

        """
        scores_ = []
        for val in self.__params_gen:
            params = {key[1]: val[key[0]] for key in enumerate(self._params_range.keys())}
            scores_.append(self.__get_scores(params, n))
        self.__get_stats(scores_)

        opt_params = list(self._params_stats[0].keys())[self._params_stats[1]]
        return dict(zip(self._params_range.keys(), opt_params))
