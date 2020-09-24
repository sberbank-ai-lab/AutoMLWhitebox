import re
import collections

from collections import namedtuple
from typing import Dict, Iterable, Hashable, Union

import numpy as np
import pandas as pd

np.random.seed(42)

Result = namedtuple("Result", ["score", "reg_alpha", "is_neg", "min_weights"])

# (Значение метрики, значение коэффициента регуляризации, отрицательные ли веса модели, веса модели)


def bootstrap(data: pd.DataFrame, iteration: int = 300):
    """
    Bootstrap of the data in range iterations

    Parameters
    ----------
    data
        DataFrame with 2 columns: 1st - feature 2nd - target iteration - number of iterations for bootstrap
    iteration

    Returns
    -------

    DataFrame with calculated WOE for each label for each bootstrap iteration
    """

    def ctab(df: pd.DataFrame, iter_num: int) -> pd.DataFrame:
        """
        Technical function. Calculates WOE for bootstrap

        Parameters
        ----------
        df
        iter_num
            num of iteration

        Returns
        -------

        cat	   "a"	        "b"	        "c"	        "d"
        177	0.276623	-0.226741	-0.792647	-0.638074


        """
        col = df.columns
        ctab_ = pd.crosstab(index=df[col[0]], columns=df[col[1]])
        ctab_.columns = ["0", "1"]

        ctab_["0"] = (ctab_["0"] + 0.5) / (1.0 * ctab_["0"].sum() + 0.5)
        ctab_["1"] = (ctab_["1"] + 0.5) / (1.0 * ctab_["1"].sum() + 0.5)
        ctab_["woe"] = np.log(ctab_["0"] / (1.0 * ctab_["1"]))
        ctab_["pctn"] = (ctab_["0"] + ctab_["1"]) / ctab_.sum()
        df_ = pd.DataFrame(ctab_["woe"]).T
        df_.index = [iter_num]
        return df_

    s = np.random.randint(0, data.shape[0], data.shape[0])

    data_ = (data.iloc[s]).copy()
    df_boot = ctab(df=data_, iter_num=0)

    for i in range(1, iteration):
        s = np.random.randint(0, data.shape[0], data.shape[0])
        data_ = (data.iloc[s]).copy()
        df_boot = pd.concat([df_boot, ctab(df=data_, iter_num=i)])

    return df_boot


def code(x: Union[float, int], text: str) -> str:
    """
    Вспомогательная функция для кодирования бинов

    Parameters
    ----------
    x

    text: str

    Returns
    -------

    """
    return str(float(x)) + "__" + text


def drop_keys(dict_: Dict, keys: Iterable[Hashable]) -> Dict:
    """
    Dropping a few keys

    Parameters
    ----------
    dict_
    keys

    Returns
    -------

    """
    for key in keys:
        dict_.pop(key)
    return dict_




def flatten(d: Dict, parent_key='', sep='_'):
    """
    Parameters
    ----------
    d: Dict

    parent_key:

    sep:

    Returns
    -------

    """
    items = []
    for k, v in d.items():
        new_key = parent_key + sep + k if parent_key else k
        if isinstance(v, collections.MutableMapping):
            items.extend(flatten(v, new_key, sep=sep).items())
        else:
            items.append((new_key, v))
    return dict(items)
