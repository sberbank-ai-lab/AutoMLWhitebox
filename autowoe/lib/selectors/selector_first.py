"""Selection of features according to the importance of the model."""

import logging

from copy import deepcopy
from typing import Any
from typing import Dict
from typing import Hashable
from typing import Optional
from typing import Tuple
from typing import Union

import lightgbm as lgb
import numpy as np
import pandas as pd

from pandas import DataFrame
from sklearn.metrics import mean_squared_error
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split

from autowoe.lib.utilities.eli5_permutation import get_score_importances

from ..logging import get_logger
from ..utilities.utils import TaskType
from ..utilities.utils import drop_keys


pd.options.mode.chained_assignment = None

logger = get_logger(__name__)

root_logger = logging.getLogger()
level = root_logger.getEffectiveLevel()

if level in (logging.CRITICAL, logging.ERROR, logging.WARNING):
    verbose_eval = 0  # False
elif level == logging.INFO:
    verbose_eval = 100
else:
    verbose_eval = 10


def nan_constant_selector(
    data: DataFrame, features_type: Dict[Hashable, str], th_const: Union[int, float] = 32
) -> Tuple[DataFrame, Dict[Hashable, str]]:
    """Selector NaN / Const columns.

    Filters columns with a large number of NaN-values or with almost constant values.

    Args:
        data: DataFrame
        features_type: Dict[Hashable, str]
        th_const: Constant threshold. Filters if the number of valid values is less than the threshold.

    Returns:
        Data, features list.

    """
    th_ = data.shape[0] - th_const

    features_to_drop = []

    for col in features_type:
        nan_count = data[col].isna().sum()
        if nan_count >= th_:
            features_to_drop.append(col)
        else:
            vc = data[col].value_counts().values[0]
            if vc >= th_:
                features_to_drop.append(col)

    logger.info(f" features {features_to_drop} contain too many nans or identical values")
    data.drop(columns=features_to_drop, axis=1, inplace=True)
    features_type = drop_keys(features_type, features_to_drop)
    return data, features_type


def get_score_function(model, task: TaskType):
    """Score function for task - {BIN: ROC_AUC, REG: MSE}."""
    if task == TaskType.BIN:
        return lambda x, y: roc_auc_score(y, model.predict_proba(x)[:, 1])
    else:
        return lambda x, y: -mean_squared_error(y, model.predict(x))


def feature_imp_selector(
    data: DataFrame,
    task: TaskType,
    features_type: Dict[Hashable, str],
    features_mark_values: Optional[Dict[str, Tuple[Any]]],
    target_name: Hashable,
    imp_th: float,
    imp_type: str,
    select_type: Union[None, int],
    process_num: int,
) -> Tuple[DataFrame, Dict[Hashable, str]]:
    """Features selection by imp_type.

    Available FS:
        - lgbm feature_importance
        - permutation importance

    Args:
        data: Dataset.
        task: Task.
        features_type: Features types.
        features_mark_values: Marked values of feature.
        target_name: Target column name.
        imp_th: Importance threshold.
        imp_type: Importance type ("feature_imp" -- feature_importances, "perm_imp" -- permutation_importances).
        select_type: Type of first feature selection.
            - If `None` then choose feautures with `feature_importance > 0`.
            - If `int` then choose the N-th best features.
        process_num: Number of threads.

    Returns:
        Data, features.

    """
    data_ = deepcopy(data)

    if features_mark_values:
        for col, mvs in features_mark_values.items():
            data_ = data_[~data_[col].isin(mvs)]

    categorical_feature = [key for key in features_type if features_type[key] == "cat"]
    if categorical_feature:
        data_[categorical_feature] = data_[categorical_feature].astype("category")

    train, test = train_test_split(data_, test_size=0.2, random_state=42)
    params = {
        "boosting_type": "gbdt",
        "n_jobs": process_num,
        "bagging_seed": 323,
        "min_gain_to_split": 0.01,
    }

    if task == TaskType.BIN:
        params["objective"] = "binary"
        params["metric"] = "auc"
    elif task == TaskType.REG:
        params["objective"] = "regression"
        params["metric"] = "mse"
    else:
        raise RuntimeError("Wrong task value")

    if imp_type == "feature_imp":
        lgb_train = lgb.Dataset(
            data=train.drop(target_name, axis=1), label=train[target_name], categorical_feature=categorical_feature
        )
        lgb_test = lgb.Dataset(
            data=test.drop(target_name, axis=1), label=test[target_name], categorical_feature=categorical_feature
        )

        model = lgb.train(
            params=params,
            train_set=lgb_train,
            early_stopping_rounds=10,
            valid_sets=[lgb_test],
            valid_names=["val_set"],
            verbose_eval=verbose_eval,
        )
        imp_dict = dict(zip(train.drop(target_name, axis=1).columns, model.feature_importance()))
    elif imp_type == "perm_imp":
        if task == TaskType.BIN:
            model = lgb.LGBMClassifier(**params)
        else:
            model = lgb.LGBMRegressor(**params)

        for cat in categorical_feature:
            vc = train[cat].value_counts()

            vc = vc[vc > 1]
            vc = vc + np.arange(vc.shape[0]) / vc.shape[0]
            train[cat] = train[cat].map(vc).astype(np.float32).fillna(0).values
            test[cat] = test[cat].map(vc).astype(np.float32).fillna(0).values

        test_ = test.drop(target_name, axis=1).astype(np.float32).values

        model.fit(
            X=train.drop(target_name, axis=1).astype(np.float32).values,
            y=train[target_name].values,
            eval_set=[(test_, test[target_name].values)],
            eval_names=["val_set"],
            eval_metric=params["metric"],
            early_stopping_rounds=10,
            verbose=verbose_eval,
        )
        _, score_decreases = get_score_importances(
            score_func=get_score_function(model, task), X=test_, y=test[target_name]
        )
        col = list(train.columns)
        col.remove(target_name)
        imp_dict = dict(zip(col, np.array(score_decreases).min(axis=0, initial=None)))
    else:
        raise ValueError("imp_type is feature_imp or perm_imp")

    if isinstance(select_type, int):
        features_to_drop, _ = zip(*sorted(imp_dict.items(), key=lambda x: x[1], reverse=True))
        features_to_drop = list(features_to_drop[select_type:])
    elif select_type is None:
        features_to_drop = [x for x in imp_dict if imp_dict[x] <= imp_th]
    else:
        raise ValueError("select_type is None or int > 0")
    logger.info(f" features {features_to_drop} have low importance")
    data.drop(columns=features_to_drop, axis=1, inplace=True)
    features_type = drop_keys(features_type, features_to_drop)
    return data, features_type
