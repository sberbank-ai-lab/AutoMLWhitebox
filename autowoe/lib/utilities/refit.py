from typing import Tuple, Optional, cast
from sklearn.svm import l1_min_c

from sklearn.linear_model import LogisticRegression
import numpy as np
from scipy import stats
from copy import deepcopy

from ..logger import get_logger

logger = get_logger(__name__)


def refit_reg(x_train: np.ndarray, y: np.ndarray, l1_grid_size: int, l1_exp_scale: float,
              max_penalty: float, interp: bool = True
              ) -> Tuple[np.ndarray, float, np.ndarray]:
    """
    Final model refit with regularization

    Args:
        x_train:
        y:
        l1_grid_size:
        l1_exp_scale:
        max_penalty:
        interp:

    Returns:

    """
    clf = LogisticRegression(penalty='l1', solver='saga', warm_start=True,
                             intercept_scaling=100000)
    cs = l1_min_c(x_train, y, loss='log', fit_intercept=True) * np.logspace(0, l1_exp_scale, l1_grid_size)
    cs = cs[cs <= max_penalty]
    # add final penalty
    if cs[-1] < max_penalty:
        cs = list(cs)
        cs.append(max_penalty)

    # fit path 
    weights, intercepts = [], []
    for c in cs:
        clf.set_params(C=c)
        clf.fit(x_train, y)
        weights.append(deepcopy(clf.coef_[0]))
        intercepts.append(clf.intercept_[0])

    if not interp:
        w, i = weights[-1], intercepts[-1]
        neg = w != 0
        return w[neg], i, neg

    for w, i in zip(weights[::-1], intercepts[::-1]):

        pos = (w > 0).sum()
        if pos > 0:
            continue

        neg = w < 0
        return w[neg], i, neg

    raise ValueError('No negative weights grid')


def refit_simple(x_train: np.ndarray, y: np.ndarray, interp: bool = True,
                 p_val: float = 0.05, x_val: Optional[np.ndarray] = None, y_val: Optional[np.ndarray] = None
                 ) -> Tuple[np.ndarray, float, np.ndarray, np.ndarray, np.ndarray]:
    """
    Final model refit with stat model mode

    Args:
        x_train:
        y:
        interp:
        p_val:
        x_val:
        y_val:

    Returns:

    """
    sl_ok = np.ones(x_train.shape[1], dtype=bool)

    n = -1

    while True:
        n += 1
        assert sl_ok.sum() > 0, 'No features left to fit on iter'.format(n)

        logger.info('Iter {0} of final refit starts with {1} features'.format(n, sl_ok.sum()))

        x_train_ = x_train[:, sl_ok]
        # индексы в исходном массиве
        ok_idx = np.arange(x_train.shape[1])[sl_ok]

        clf = LogisticRegression(penalty='none', solver='lbfgs', warm_start=False,
                                 intercept_scaling=1)
        clf.fit(x_train_, y)

        # check negative coefs here if interp
        sl_pos_coef = np.zeros((x_train_.shape[1],), dtype=np.bool)
        if interp:
            sl_pos_coef = clf.coef_[0] >= 0

        # если хотя бы один неотрицательный - убирай самый большой и по новой
        if sl_pos_coef.sum() > 0:
            max_coef_idx = clf.coef_[0].argmax()
            sl_ok[ok_idx[max_coef_idx]] = False
            continue

        # если прошли все отрицательные смотрим на pvalue
        p_vals, b_var = calc_p_val(x_train_, clf.coef_[0], clf.intercept_[0])
        # без интерсепта
        p_vals_f = p_vals[:-1]

        model_p_vals = p_vals.copy()
        model_b_var = b_var.copy

        # если хотя бы один больше p_val - дропай самый большой и погнали по новой
        if p_vals_f.max() > p_val:
            max_p_val_idx = p_vals_f.argmax()
            sl_ok[ok_idx[max_p_val_idx]] = False
            continue

        if x_val is not None:
            # то же самое на валидационной выборке
            logger.info('Validation data checks')
            x_val_ = x_val[:, sl_ok]

            p_vals, b_var = calc_p_val_on_valid(x_val_, y_val)
            p_vals_f = p_vals[:-1]

            # если хотя бы один больше p_val - дропай самый большой и погнали по новой
            if p_vals_f.max() > p_val:
                max_p_val_idx = p_vals_f.argmax()
                sl_ok[ok_idx[max_p_val_idx]] = False
                continue

        weights = cast(np.ndarray, clf.coef_[0])
        intercept = cast(float, clf.intercept_[0])

        return weights, intercept, sl_ok, cast(np.ndarray, model_p_vals), cast(np.ndarray, model_b_var)


def calc_p_val(x_train: np.ndarray, weights: np.ndarray, intercept: float) -> Tuple[np.ndarray, np.ndarray]:
    """
    Calc p-values for coef estimates

    Args:
        x_train:
        weights:
        intercept:

    Returns:

    """
    coef_ = np.concatenate([weights, [intercept]])
    x_train = np.concatenate([x_train, np.ones((x_train.shape[0], 1))], axis=1)
    prob_ = 1 / (1 + np.exp(-np.dot(x_train, coef_)))
    prob_ = prob_ * (1 - prob_)
    hess = np.dot((prob_[:, np.newaxis] * x_train).T, x_train)

    inv_hess = np.linalg.inv(hess)
    b_var = inv_hess.diagonal()
    w_stat = (coef_ ** 2) / b_var

    p_vals = 1 - stats.chi2(1).cdf(w_stat)

    return p_vals, b_var


def calc_p_val_on_valid(x_train, y) -> Tuple[np.ndarray, np.ndarray]:
    """
    Fit algo and calc p-values

    Args:
        x_train:
        y:

    Returns:

    """
    pv_mod = LogisticRegression(penalty='none', solver='lbfgs')
    pv_mod.fit(x_train, y)

    return calc_p_val(x_train, pv_mod.coef_[0], pv_mod.intercept_[0])
