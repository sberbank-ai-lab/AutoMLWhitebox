"""Compose several selector."""

from copy import copy
from typing import Any
from typing import Dict
from typing import List
from typing import Optional
from typing import Tuple
from typing import TypeVar

import numpy as np
import pandas as pd
import scipy as sp

from sklearn.metrics import r2_score
from sklearn.metrics import roc_auc_score

from ..logging import get_logger
from ..utilities.utils import TaskType
from ..utilities.utils import feature_changing
from .utils import F_LIST_TYPE


logger = get_logger(__name__)

WoE = TypeVar("WoE")


class ComposedSelector:
    """Compose feature selector.

    Sequential filtering of features by rules:
        1) Unique WoE value.
        2) Singel feature model has metric lower than threshold.
        3) VIF of feature greater than threshold.
        4) There are features with a pair correlation above the threshold.

    Metrics:
        1) BIN - AUC
        2) REG - R2

    Args:
        train: Train features.
        target: Train target.
        task: Task.
        features_mark_values: Marked values of features.

    """

    default_metric_th = {TaskType.BIN: 0.5, TaskType.REG: 0.0}

    def __init__(
        self,
        train: pd.DataFrame,
        target: pd.Series,
        task: TaskType,
        features_mark_values: Optional[Dict[str, Tuple[Any]]],
    ):
        self.train = train
        self.target = target
        self.task = task
        self.features_mark_values = features_mark_values
        # precompute corrs

        if features_mark_values is not None:
            mask_good_values = pd.Series([True] * train.shape[0])
            for col, mvs in features_mark_values.items():
                mask_good_values = mask_good_values & (~train[col].isin(mvs))
        else:
            mask_good_values = pd.Series([True] * train.shape[0], index=train.index)
        train_values = train[mask_good_values].values

        cc = np.abs(sp.corrcoef(train_values, rowvar=False))
        self.precomp_corr = pd.DataFrame(cc, index=train.columns, columns=train.columns)

        metrics = []
        for col in train.columns:
            if task == TaskType.BIN:
                m = 1 - roc_auc_score(target, train[col])
            else:
                m = r2_score(target, train[col])
            metrics.append(m)
        self.precomp_metrics = pd.Series(metrics, index=train.columns)

    @staticmethod
    def __compare_msg(closure, value, msg=None):
        flg = closure(value)
        if not flg:
            logger.info(msg)
        return flg

    def __call__(
        self,
        feature_history: Dict[str, str],
        features_fit: List[str],
        pearson_th: float = 0.9,
        metric_th: Optional[float] = None,
        vif_th: float = 5.0,
    ) -> F_LIST_TYPE:
        """Filtered features."""
        if metric_th is None:
            metric_th = self.default_metric_th[self.task]

        candidates = copy(features_fit)
        features_before = set(candidates)

        # откинем константные
        _, filter_features = feature_changing(
            feature_history,
            "Constant WoE value",
            features_before,
            lambda candidates: (
                None,
                [
                    col
                    for col in candidates
                    if self.__compare_msg(
                        lambda x: ~np.isnan(self.precomp_corr.loc[x, x]),
                        col,
                        "Feature {0} removed due to single WOE value".format(col),
                    )
                ],
            ),  # func
            candidates,  # args
            # ...,  # kwargs
        )

        # откинем с низкой метрикой
        _, filter_features = feature_changing(
            feature_history,
            "Low metric value",  # TODO: feature name
            filter_features,
            lambda candidates: (
                None,
                [
                    col
                    for col in candidates
                    if self.__compare_msg(
                        lambda x: self.precomp_metrics[x] >= metric_th,
                        col,
                        "Feature {0} removed due to low metric value {1}".format(col, self.precomp_metrics[col]),
                    )
                ],
            ),  # func
            filter_features,  # args
            # ...,  # kwargs
        )
        candidates = filter_features

        # итеративный виф
        max_vif = np.inf
        while max_vif > vif_th:
            corrs = self.precomp_corr.loc[candidates, candidates]
            # fix singularity
            corrs = corrs.values + np.diag(np.ones(corrs.shape[0]) * 1e-4)
            vifs = np.linalg.inv(corrs).diagonal()

            max_vif_idx = vifs.argmax()
            max_vif = vifs[max_vif_idx]

            if max_vif >= vif_th:
                logger.info("Feature {0} removed due to high VIF value = {1}".format(candidates[max_vif_idx], max_vif))
                if feature_history is not None:
                    feature_history[candidates[max_vif_idx]] = f"High VIF value = {round(max_vif, 2)}"
                candidates = [x for (n, x) in enumerate(candidates) if n != max_vif_idx]

                # попарные корреляции
        # отсортируем по убыванию метрики
        order_ = np.array([self.precomp_metrics[x] for x in candidates]).argsort()[::-1]
        candidates = [candidates[x] for x in order_]

        n = 0
        while n < (len(candidates) - 1):

            partial_corrs = self.precomp_corr.loc[candidates[n], candidates[n + 1 :]]
            big_partial_corrs = partial_corrs[partial_corrs >= pearson_th]
            if len(big_partial_corrs) > 0:
                logger.info(
                    "Features {0}: metric = {1} was removed due to corr = {2} with feat {3}: metric = {4}".format(
                        list(big_partial_corrs.index.values),
                        list(self.precomp_metrics[big_partial_corrs.index]),
                        list(big_partial_corrs.values),
                        candidates[n],
                        self.precomp_metrics[candidates[n]],
                    )
                )
                if feature_history is not None:
                    for feat in big_partial_corrs.index.values:
                        feature_history[feat] = f"High correlation with feat {candidates[n]}"

            candidates = [x for x in candidates if x not in set(big_partial_corrs.index.values)]
            n += 1

        return candidates
