import lightgbm as lgb
import numpy as np

from copy import deepcopy
from typing import Union, Dict, List, TypeVar, Tuple, Hashable

from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split

from autowoe.lib.utilities.eli5_permutation import get_score_importances

from ..utilities.utilities import drop_keys

DataFrame = TypeVar("DataFrame")
feature = Union[str, int, float]
f_list_type = List[feature]


def nan_constant_selector(data: DataFrame, features_type: Dict[Hashable, str],
                          th_const: Union[int, float] = 32) -> Tuple[DataFrame, Dict[Hashable, str]]:
    """
    Отбор колонок, в которых много нанов или почти константных колонок

    Parameters
    ----------
    data: DataFrame

    features_type: Dict[Hashable, str]

    th_const: int, float
        Если число валидных значений больше трешхолда, то колонка не константная (int)
        В случае указания float, число валидных значений будет определяться как размер_выборки * th_const
    Returns
    -------

    """
    if isinstance(th_const, float):
        th_const = int(th_const * data.shape[0])
        
    th_ = data.shape[0] - th_const
    
    features_to_drop = []
    
    for col in features_type:
        nan_count = data[col].isna().sum()
        if nan_count >= th_:
            features_to_drop.append(col)
#         vc = data[col].value_counts().values[0]
#         if vc > th_:
#             features_to_drop.append(col)
        # фикс от Антона)
        else:
            vc = data[col].value_counts().values[0]
            if vc >= th_:
                features_to_drop.append(col)
                
    print(f" features {features_to_drop} contain too many nans or identical values")
    data.drop(columns=features_to_drop, axis=1, inplace=True)
    features_type = drop_keys(features_type, features_to_drop)
    return data, features_type


def feature_imp_selector(data: DataFrame, features_type: Dict[Hashable, str], target_name: Hashable, imp_th: float,
                         imp_type: str, select_type: Union[None, int],
                         process_num: int) -> Tuple[DataFrame, Dict[Hashable, str]]:
    """
    Parameters
    ----------
    data: DataFrame

    features_type:

    target_name:

    imp_th:

    imp_type: str
        Способ измерения важности признаков. "feature_imp" -- feature_importances, "perm_imp" -- permutation_importances

    select_type:
        Тип первичного отбора прищнаков, если число, то
        оставляем только столько признаков (самых лучших по feature_importance).
        Если None оставлям те, у которых feature_importance больше 0.

    process_num:


    Returns
    -------
    """

    data_ = deepcopy(data)

    categorical_feature = [key for key in features_type if features_type[key] == "cat"]
    if categorical_feature:
        data_[categorical_feature] = data_[categorical_feature].astype("category")

    train, test = train_test_split(data_, test_size=0.2, random_state=42)
    params = {
        "boosting_type": "gbdt",
        "objective": "binary",
        "n_jobs": process_num,
        "bagging_seed": 323,
        "metric": "auc",
        "min_gain_to_split": 0.01,
        # "learning_rate": 0.05,
        # "verbosity": -1
    }

    if imp_type == "feature_imp":
        lgb_train = lgb.Dataset(data=train.drop(target_name, axis=1),
                                label=train[target_name],
                                categorical_feature=categorical_feature
                               )
        lgb_test = lgb.Dataset(data=test.drop(target_name, axis=1),
                               label=test[target_name],
                               categorical_feature=categorical_feature
                              )

        clf = lgb.train(params=params, train_set=lgb_train, verbose_eval=1000,
                        early_stopping_rounds=10, valid_sets=[lgb_test], valid_names=['val_set'])
        imp_dict = dict(zip(train.drop(target_name, axis=1).columns, clf.feature_importance()))
    elif imp_type == "perm_imp":
        clf = lgb.LGBMClassifier(**params)

        for cat in categorical_feature:
            vc = train[cat].value_counts()
            # Фикс от Антона)
            vc = vc[vc > 1]
            vc = vc + np.arange(vc.shape[0]) / vc.shape[0]
            train.loc[:, cat] = train[cat].map(vc).astype(np.float32).fillna(0).values
            test.loc[:, cat] = test[cat].map(vc).astype(np.float32).fillna(0).values
        # здесь тоже от Антона
        test_ = test.drop(target_name, axis=1).astype(np.float32).values
        
        clf.fit(X=train.drop(target_name, axis=1).astype(np.float32).values, y=train[target_name].values,
                eval_set=[(test_, test[target_name].values)],
                eval_names=['val_set'], eval_metric='auc', early_stopping_rounds=10,
                verbose=1000
               )
        _, score_decreases = get_score_importances(score_func=lambda x, y: roc_auc_score(y, clf.predict_proba(x)[:, 1]),
                                                   X=test_,
                                                   y=test[target_name])
        col = list(train.columns)
        col.remove(target_name)
        imp_dict = dict(zip(col, np.array(score_decreases).min(axis=0)))
        
    else:
        raise ValueError("imp_type is feature_imp or perm_imp")

    if isinstance(select_type, int):
        features_to_drop, _ = zip(*sorted(imp_dict.items(), key=lambda x: x[1], reverse=True))
        features_to_drop = list(features_to_drop[select_type:])
    elif select_type is None:
        features_to_drop = [x for x in imp_dict if imp_dict[x] <= imp_th]
    else:
        raise ValueError("select_type is None or int > 0")
    print(f" features {features_to_drop} have low importance")
    data.drop(columns=features_to_drop, axis=1, inplace=True)
    features_type = drop_keys(features_type, features_to_drop)
    return data, features_type

