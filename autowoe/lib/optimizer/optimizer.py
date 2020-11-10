from copy import copy
import lightgbm as lgb
import pandas as pd
import numpy as np

from typing import Any, Dict, Iterable, List, Tuple, Union
from collections import OrderedDict
from itertools import product

# TODO: Do we need random state here?
np.random.seed(232)


class TreeParamOptimizer:
    """
    Класс для оптимизации гиперпараметров решающего дерева
    """

    def __init__(self, data: pd.DataFrame, params_range: Dict[str, tuple], n_folds: int = 5):
        """

        Args:
            data: Данные с признаком для которого производится биннинг (I колонка) и target (II колонка)
            params_range: OrderedDict with parameters and ranges for binning algorithms
                Ex. params_range = OrderedDict({"max_depth": (4, 7, 17, 2, 3),  "min_child_samples": (40000, 20000, 5000),})
        """
        self._params_range = copy(params_range)
        ds_params = {}
        try:
            ds_params['min_data_in_bin'] = self._params_range.pop('min_data_in_bin')[0]
        except KeyError:
            pass

        self._lgb_train = lgb.Dataset(data=pd.DataFrame(data.iloc[:, 0]), label=data.iloc[:, 1], params=ds_params)
        self._params_stats = None
        self.n_folds = n_folds

    @property
    def __params_gen(self) -> Iterable[Tuple]:
        """

        Returns:

        """
        return product(*self._params_range.values())

    def _get_score(self, params: Dict[str, Any], n: int) -> List[float]:
        """

        Args:
            params: параметры дерева (которые мы ищем в этом классе)
            n: количество кросс-валидаций для оценки гиперпараметров

        Returns:

        """
        default_tree_params = {
            "boosting_type": "gbdt",
            "learning_rate": 1,
            "objective": "binary",
            "bagging_freq": 1,
            "bagging_fraction": 1,
            "feature_fraction": 1,
            "bagging_seed": 323,
            "n_jobs": 1,
            "verbosity": -1}
        unite_params = {**params, **default_tree_params}

        score = []
        for seed in range(n):
            cv_results = lgb.cv(params=unite_params, train_set=self._lgb_train, num_boost_round=1,
                                nfold=self.n_folds, metrics='auc',
                                stratified=True, shuffle=True, seed=seed)
            score.append(cv_results["auc-mean"])
        return score

    def __get_stats(self, stats: List[List[float]]):
        """
        Функция вычисления статистик по scores
        Присутствуют три измерения. 0 - число комбинаций параметров, 1 - число кросс-валидаций, 2 - число фолдов к cv

        Args:
            stats:

        Returns:

        """
        stats = np.array(stats)
        median_, std_ = np.median(stats, axis=(1, 2)), np.std(stats, axis=(1, 2))

        id_max = zip(*(median_, -std_))  # номер комбинации с наилучшим качеством
        id_max = max(enumerate(id_max), key=lambda x: x[1])[0]

        stat_score = zip(*(median_, std_))
        self._params_stats = OrderedDict((key, value) for (key, value) in zip(self.__params_gen, stat_score)), id_max

    def __call__(self, n: int) -> Dict[str, Union[int, str, None]]:
        """

        Args:
            n: Число перебираемых гиперпараметров

        Returns:
            Параметры с наилучшим качеством (roc_auc) классификации
        """

        score_ = []
        for val in self.__params_gen:
            params = {key[1]: val[key[0]] for key in enumerate(self._params_range.keys())}
            score_.append(self._get_score(params, n))
        self.__get_stats(score_)

        opt_params = list(self._params_stats[0].keys())[self._params_stats[1]]
        return dict(zip(self._params_range.keys(), opt_params))
