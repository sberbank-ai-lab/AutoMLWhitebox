"""Additional functional for refitting model."""

from copy import deepcopy
from typing import Optional
from typing import Tuple
from typing import cast

import numpy as np

from scipy import linalg
from scipy import stats
from sklearn.linear_model import Lasso
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import LogisticRegression
from sklearn.svm import l1_min_c

from ..logging import get_logger
from .utils import TaskType


logger = get_logger(__name__)


def refit_reg(
    task: TaskType,
    x_train: np.ndarray,
    y: np.ndarray,
    l1_grid_size: int,
    l1_exp_scale: float,
    max_penalty: float,
    interp: bool = True,
) -> Tuple[np.ndarray, float, np.ndarray]:
    """Final model refit with regularization.

    Args:
        task: Task.
        x_train: Train features.
        y: Train target.
        l1_grid_size: Number of point at regularized grid.
        l1_exp_scale: Maximum value of `C` coefficient.
        max_penalty: maximum value of `C` coefficient.
        interp: Interpreted model.

    Returns:
        Weights , intercept of model, features mask.

    """
    weights, intercepts = [], []
    if task == TaskType.BIN:
        clf = LogisticRegression(penalty="l1", solver="saga", warm_start=True, intercept_scaling=100000)
        cs = l1_min_c(x_train, y, loss="log", fit_intercept=True) * np.logspace(0, l1_exp_scale, l1_grid_size)
        cs = cs[cs <= max_penalty]
        # add final penalty
        if cs[-1] < max_penalty:
            cs = list(cs)
            cs.append(max_penalty)

        # fit path
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
    else:
        cs_max_penalty = 1 / max_penalty
        model = Lasso(warm_start=True, positive=interp)
        cs = np.logspace(0, l1_exp_scale, l1_grid_size + 1)
        cs = cs[cs <= cs_max_penalty]
        # add final penalty
        if cs[-1] < cs_max_penalty:
            cs = list(cs)
            cs.append(cs_max_penalty)
        cs = np.array(cs)

        alphas = (1.0 / cs[1:])[::-1]

        for alpha in alphas:
            model.set_params(alpha=alpha)
            model.fit(x_train, y)
            weights.append(model.coef_)
            intercepts.append(model.intercept_)

        w, i = weights[0], intercepts[0]
        pos = w >= 0

        return w[pos], i, pos

    raise ValueError("No negative weights grid")


def refit_simple(
    task: TaskType,
    x_train: np.ndarray,
    y: np.ndarray,
    interp: bool = True,
    p_val: float = 0.05,
    x_val: Optional[np.ndarray] = None,
    y_val: Optional[np.ndarray] = None,
    n_jobs: int = -1,
) -> Tuple[np.ndarray, float, np.ndarray, np.ndarray, np.ndarray]:
    """Final model refit with stat model mode.

    Args:
        task: Task.
        x_train: Train features.
        y: Train target.
        interp: Intepreted model.
        p_val: P-value.
        x_val: Validation features.
        y_val: Validation target.
        n_jobs: Number of threads.

    Returns:
        weights, intercept, features mask, p values, b vars.

    """
    sl_ok = np.ones(x_train.shape[1], dtype=bool)

    n = -1

    while True:
        n += 1
        assert sl_ok.sum() > 0, "No features left to fit on iter"

        logger.info("Iter {0} of final refit starts with {1} features".format(n, sl_ok.sum()))

        x_train_ = x_train[:, sl_ok]
        # индексы в исходном массиве
        ok_idx = np.arange(x_train.shape[1])[sl_ok]

        if task == TaskType.BIN:
            model = LogisticRegression(penalty="none", solver="lbfgs", warm_start=False, intercept_scaling=1)
            model.fit(x_train_, y)
            model_coef = model.coef_[0]
            model_intercept = model.intercept_[0]
        else:
            model = LinearRegression(n_jobs=n_jobs)
            model.fit(x_train_, y)
            model_coef = model.coef_
            model_intercept = model.intercept_

        # check negative coefs here if interp
        sl_pos_coef = np.zeros((x_train_.shape[1],), dtype=np.bool)
        if interp:
            sl_pos_coef = model.coef_[0] >= 0 if task == TaskType.BIN else model.coef_[0] <= 0

        # если хотя бы один неотрицательный - убирай самый большой и по новой
        if sl_pos_coef.sum() > 0:
            max_coef_idx = model_coef.argmax()
            sl_ok[ok_idx[max_coef_idx]] = False
            continue

        # если прошли все отрицательные смотрим на pvalue
        if task == TaskType.BIN:
            p_vals, b_var = calc_p_val(x_train_, model_coef, model_intercept)
        else:
            p_vals, b_var = calc_p_val_reg(x_train_, y, model_coef, model_intercept)

        # без интерсепта
        p_vals_f = p_vals[:-1]

        model_p_vals = p_vals.copy()
        model_b_var = b_var.copy() if b_var is not None else None

        # если хотя бы один больше p_val - дропай самый большой и погнали по новой
        if p_vals_f.max() > p_val:
            max_p_val_idx = p_vals_f.argmax()
            sl_ok[ok_idx[max_p_val_idx]] = False
            continue

        if x_val is not None:
            # то же самое на валидационной выборке
            logger.info("Validation data checks")
            x_val_ = x_val[:, sl_ok]

            p_vals, b_var = calc_p_val_on_valid(x_val_, y_val, task, n_jobs)
            p_vals_f = p_vals[:-1]

            # если хотя бы один больше p_val - дропай самый большой и погнали по новой
            if p_vals_f.max() > p_val:
                max_p_val_idx = p_vals_f.argmax()
                sl_ok[ok_idx[max_p_val_idx]] = False
                continue

        weights = cast(np.ndarray, model_coef)
        intercept = cast(float, model_intercept)

        return weights, intercept, sl_ok, cast(np.ndarray, model_p_vals), cast(np.ndarray, model_b_var)


def calc_p_val(x_train: np.ndarray, weights: np.ndarray, intercept: float) -> Tuple[np.ndarray, np.ndarray]:
    """Calc p-values for coef estimates.

    Args:
        x_train: Train features.
        weights: Model Weights.
        intercept: Model intercept coefficient.

    Returns:
        p values, b vars.

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


def calc_p_val_on_valid(
    x_train: np.ndarray, y: np.ndarray, task: TaskType, n_jobs: int = -1
) -> Tuple[np.ndarray, np.ndarray]:
    """Fit algo and calc p-values.

    Args:
        x_train: Train features.
        y: Train target.
        task: Task.
        n_jobs: Number of threads.

    Returns:
        p values, b vars.

    """
    if task == TaskType.BIN:
        model = LogisticRegression(penalty="none", solver="lbfgs", warm_start=False, intercept_scaling=1)
        model.fit(x_train, y)

        return calc_p_val(x_train, model.coef_[0], model.intercept_[0])
    else:
        model = LinearRegression(n_jobs=n_jobs)
        model.fit(x_train, y)

        return calc_p_val_reg(x_train, y, model.coef_, model.intercept_)


def calc_p_val_reg(
    x_train: np.ndarray, y_train: np.ndarray, weights: np.ndarray, intercept: float
) -> Tuple[np.ndarray, np.ndarray]:
    """Calculate p values for regression task."""
    n, k = x_train.shape
    y_pred = (np.dot(x_train, weights) + intercept).T

    # Change X and Y into numpy matricies. x also has a column of ones added to it.
    x = np.hstack((np.matrix(x_train), np.ones((n, 1))))
    y_train = np.matrix(y_train).T

    # Degrees of freedom.
    df = float(n - k - 1)

    # Sample variance.
    sse = np.sum(np.square(y_pred - y_train), axis=0)
    sampleVariance = sse / df

    # Sample variance for x.
    sampleVarianceX = x.T * x

    # Covariance Matrix = [(s^2)(X'X)^-1]^0.5. (sqrtm = matrix square root.  ugly)
    covarianceMatrix = linalg.sqrtm(sampleVariance[0, 0] * sampleVarianceX.I)

    # Standard erros for the difference coefficients: the diagonal elements of the covariance matrix.
    se = covarianceMatrix.diagonal()  # [1:]

    # T statistic for each beta.
    betasTStat = np.zeros(len(se))
    for i in range(len(se) - 1):
        betasTStat[i] = weights[i] / se[i]
    betasTStat[-1] = intercept / se[-1]

    # P-value for each beta. This is a two sided t-test, since the betas can be
    # positive or negative.
    betasPValue = 1 - stats.t.cdf(abs(betasTStat), df)

    return betasPValue, None
