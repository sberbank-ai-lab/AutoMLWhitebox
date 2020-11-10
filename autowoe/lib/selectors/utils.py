from collections import namedtuple
from typing import Sequence, Tuple, Dict, List, Union

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegressionCV
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import BaseCrossValidator
from sklearn.svm import l1_min_c

from ..logger import get_logger

logger = get_logger(__name__)

Result = namedtuple("Result", ["score", "reg_alpha", "is_neg", "min_weights"])
feature = Union[str, int, float]
f_list_type = List[feature]


def scorer(estimator, x_train, y):
    return roc_auc_score(y, estimator.predict_proba(x_train)[:, 1])


class PredefinedFolds(BaseCrossValidator):

    def __init__(self, cv_split: Dict[int, Tuple[Sequence[int], Sequence[int]]]):
        """

        Args:
            cv_split:
        """
        self.cv_split = cv_split

    def _iter_test_indices(self, x_train: np.ndarray = None, y: np.ndarray = None, groups: np.ndarray = None) -> np.ndarray:
        """
        Generates integer indices corresponding to test sets

        Args:
            x_train:
            y:
            groups:

        Returns:

        """
        for n in self.cv_split:
            yield self.cv_split[n][1]

    def get_n_splits(self, *args, **kwargs) -> int:
        return len(self.cv_split)


def analyze_result(clf: LogisticRegressionCV, features_names: Sequence[str],
                   interpreted_model: bool = True) -> List[Result]:
    """

    Args:
        clf:
        features_names:
        interpreted_model:

    Returns:

    """
    scores = clf.scores_[1]
    cs_scores = scores.mean(axis=0)

    cs_len = scores.shape[1]
    coef_ = np.moveaxis(clf.coefs_paths_[1][:, :, :-1], 1, 0)

    if interpreted_model:
        cs_negs = (coef_.reshape((cs_len, -1)) <= 0).all(axis=1)
    else:
        cs_negs = [True] * cs_len

    cs_min_weights = [pd.Series(coef_[x].min(axis=0), index=features_names)  # .sort_values()
                      for x in range(cs_len)]

    results = [Result(score, c, is_neg, min_weights) for (score, c, is_neg, min_weights) in
               zip(cs_scores, clf.Cs, cs_negs, cs_min_weights)]

    return results


def l1_select(interpreted_model: bool,
              n_jobs: int,
              dataset: Tuple[pd.DataFrame, pd.Series],
              l1_grid_size: int,
              l1_exp_scale: float,
              cv_split: Dict[int, Tuple[Sequence[int], Sequence[int]]],
              auc_tol: float = 1e-4
              ) -> Tuple[f_list_type, Result]:
    """
    
    Args:
        interpreted_model: 
        n_jobs: 
        dataset: 
        l1_grid_size: 
        l1_exp_scale: 
        cv_split: 
        auc_tol: 

    Returns:

    """
    # get grid for cs
    cs = l1_min_c(dataset[0], dataset[1], loss='log', fit_intercept=True) * np.logspace(0, l1_exp_scale, l1_grid_size)
    logger.info('C parameter range in [{0}:{1}], {2} values'.format(cs[0], cs[-1], l1_grid_size))
    # fit model with crossvalidation
    cv = PredefinedFolds(cv_split)
    clf = LogisticRegressionCV(Cs=cs,
                               solver='saga',
                               tol=1e-5,
                               cv=cv,
                               penalty='l1',
                               scoring=scorer,
                               intercept_scaling=10000.,
                               max_iter=1000,
                               n_jobs=n_jobs,
                               random_state=42)

    clf.fit(dataset[0].values, dataset[1].values)

    # analyze cv results
    result = analyze_result(clf, dataset[0].columns, interpreted_model)

    # perform selection
    # filter bad weights models
    scores_neg = [x for x in result if x.is_neg]
    # get top score from avail models
    max_score = max([x.score for x in result])
    # get score with tolerance
    ok_score = max_score - auc_tol
    # select first model that is ok with tolerance
    res = None
    for res in scores_neg:
        if res.score >= ok_score:
            break

    # get selected features
    features_fit = [x for (x, y) in zip(dataset[0].columns, res.min_weights) if y != 0]
    logger.info(res)

    return features_fit, res
