# noqa: D100

from collections import namedtuple
from typing import List
from typing import Mapping
from typing import Sequence
from typing import Tuple
from typing import Union

import numpy as np
import pandas as pd

from sklearn.linear_model import LassoCV
from sklearn.linear_model import LogisticRegressionCV
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import BaseCrossValidator
from sklearn.svm import l1_min_c

from autowoe.lib.utilities.utils import TaskType

from ..logging import get_logger


logger = get_logger(__name__)

Result = namedtuple("Result", ["score", "reg_alpha", "is_neg", "min_weights"])

FEATURE = Union[str, int, float]
F_LIST_TYPE = Sequence[FEATURE]


def scorer(estimator, x_train, y):
    """Evaluate ROC-AUC."""
    return roc_auc_score(y, estimator.predict_proba(x_train)[:, 1])


class PredefinedFolds(BaseCrossValidator):
    """Predefined Folds."""

    def __init__(self, cv_split: Mapping[int, Tuple[Sequence[int], Sequence[int]]]):
        self.cv_split = cv_split

    def _iter_test_indices(
        self, x_train: np.ndarray = None, y: np.ndarray = None, groups: np.ndarray = None
    ) -> np.ndarray:
        """Generates integer indices corresponding to test sets.

        Args:
            x_train: Train features.
            y: Train target.
            groups: Groups.

        Yields:
            test set indexes.

        """
        for n in self.cv_split:
            yield self.cv_split[n][1]

    def get_n_splits(self, *args, **kwargs) -> int:
        """Number of splits."""
        return len(self.cv_split)


def analyze_result(
    model: Union[LogisticRegressionCV, LassoCV], features_names: Sequence[str], interpreted_model: bool = True
) -> List[Result]:
    """Analyze the result of the searching coefficient regularization.

    Args:
        model: Linear model.
        features_names: List of features names.
        interpreted_model: Build interpreted model.

    Returns:
        Summary.

    """
    scores = model.scores_[1]
    cs_scores = scores.mean(axis=0)

    cs_len = scores.shape[1]
    coef_ = np.moveaxis(model.coefs_paths_[1][:, :, :-1], 1, 0)

    if interpreted_model:
        cs_negs = (coef_.reshape((cs_len, -1)) <= 0).all(axis=1)
    else:
        cs_negs = [True] * cs_len

    cs_min_weights = [pd.Series(coef_[x].min(axis=0), index=features_names) for x in range(cs_len)]  # .sort_values()

    results = [
        Result(score, c, is_neg, min_weights)
        for (score, c, is_neg, min_weights) in zip(cs_scores, model.Cs, cs_negs, cs_min_weights)
    ]

    return results


def l1_select(
    task: TaskType,
    interpreted_model: bool,
    n_jobs: int,
    dataset: Tuple[pd.DataFrame, pd.Series],
    l1_grid_size: int,
    l1_exp_scale: float,
    cv_split: Mapping[int, Tuple[Sequence[int], Sequence[int]]],
    metric_tol: float = 1e-4,
) -> Tuple[F_LIST_TYPE, Result]:
    """Select the main features according to the lasso model.

    Args:
        task: Task.
        interpreted_model: Create interpreted model.
        n_jobs: Number of threads.
        dataset: Tuple of features and target.
        l1_grid_size: Number of points on grid.
        l1_exp_scale: Maximun value of `C`.
        cv_split: Cross-Val splits.
        metric_tol: Metric tolerance.

    Returns:
        Selected features, summary info.

    """
    # fit model with crossvalidation
    cv = PredefinedFolds(cv_split)
    if task == TaskType.BIN:
        # get grid for cs
        cs = l1_min_c(dataset[0], dataset[1], loss="log", fit_intercept=True) * np.logspace(
            0, l1_exp_scale, l1_grid_size
        )
        logger.info("C parameter range in [{0}:{1}], {2} values".format(cs[0], cs[-1], l1_grid_size))

        model = LogisticRegressionCV(
            Cs=cs,
            solver="saga",
            tol=1e-5,
            cv=cv,
            penalty="l1",
            scoring=scorer,
            intercept_scaling=10000.0,
            max_iter=1000,
            n_jobs=n_jobs,
            random_state=42,
        )
    else:
        # get grid for cs
        cs = np.logspace(0, l1_exp_scale, l1_grid_size + 1)
        alphas = 1.0 / cs[1:][::-1]
        logger.info("Alphas parameter range in [{0}:{1}], {2} values".format(alphas[0], alphas[-1], l1_grid_size))

        model = LassoCV(
            alphas=alphas, cv=cv, positive=interpreted_model, tol=1e-5, max_iter=1000, n_jobs=n_jobs, random_state=42
        )

    model.fit(dataset[0].values, dataset[1].values)

    features_fit: List[str]
    if task == TaskType.BIN:
        # analyze cv results
        result = analyze_result(model, dataset[0].columns, interpreted_model)

        # perform selection
        # filter bad weights models
        scores_neg = [x for x in result if x.is_neg]
        # get top score from avail models
        max_score = max([x.score for x in result])
        # get score with tolerance
        ok_score = max_score - metric_tol
        # select first model that is ok with tolerance
        res = None
        for res in scores_neg:
            if res.score >= ok_score:
                break

        # get selected features
        features_fit = [x for (x, y) in zip(dataset[0].columns, res.min_weights) if y != 0]
        logger.info(res)
    else:
        features_fit = [x for (x, y) in zip(dataset[0].columns, model.coef_) if y != 0]
        res = Result(
            score=model.mse_path_.mean(axis=1).min(),
            reg_alpha=model.alpha_,
            is_neg=[True] * model.coef_.shape[0],
            min_weights=np.min(model.coef_),
        )

    return features_fit, res
