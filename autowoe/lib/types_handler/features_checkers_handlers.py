import pandas as pd
import numpy as np

from typing import Optional, Tuple

F_UNIQUE = 5


def dates_checker(feature: pd.Series) -> bool:
    try:
        feature = pd.to_datetime(feature)
        if (feature.min().year <= 1975) or (feature.min().year is np.nan):
            return False
        else:
            return True
    except ValueError:
        return False
    except Exception:
        raise ValueError("Something is wrong with object types")


def dates_handler(feature: pd.Series,
                  feature_type: Tuple[Optional[str], Tuple[str]] = (None, ("wd", "m", "y", "d"))) -> Tuple:
    """
    feature_type ("%Y%d%m", ("m", "d", "wd", "h", "min")), (None, ("m", "d", "wd", "h", "min"))

    Parameters
    ----------
    feature:
        Колонка для парсинга
    feature_type:
    Returns
    -------

    """
    date_format = feature_type[0]
    seasonality = feature_type[1]

    if not len(seasonality):
        raise ValueError("Seasonality is empty!")

    # trig_name2func = {"sin": lambda x: np.sin(math.pi * x), "cos": lambda x: np.cos(math.pi * x)}

    seas2func = {
        "y": lambda x: x.year,
        "m": lambda x: x.month,  # (x.month - 1) / 11,
        "d": lambda x: x.day,  # (min(28, x.day) - 1) / 27,
        "wd": lambda x: x.weekday(),  # x.weekday() / 6,
        "h": lambda x: x.hour,  # x.hour / 23,
        "min": lambda x: x.minute  # x.minute / 59
    }

    new_features = []
    new_feature = pd.to_datetime(feature, format=date_format)

    for seas in seasonality:
        new_feature_name = str(new_feature.name) + "__F__" + seas
        # new_feature_sin = new_feature.map(lambda x: trig_name2func["sin"](seas2func[seas](x)))
        # new_feature_cos = new_feature.map(lambda x: trig_name2func["cos"](seas2func[seas](x)))
        # new_features.append((new_feature_name + "__sin", new_feature_sin))
        # new_features.append((new_feature_name + "__cos", new_feature_cos))
        new_feature_ = new_feature.map(lambda x: seas2func[seas](x))
        new_features.append((new_feature_name, new_feature_))

    return new_features, feature_type


def cat_checker(feature: pd.Series) -> bool:
    """
    Парсер категориальных признаков

    Parameters
    ----------
    feature

    Returns
    -------

    """
    if feature.dtype in [object, str, np.str]:
        return True

    # feature_unique = feature.dropna().unique()
    feature_unique = feature.unique()
    if 2 < feature_unique.shape[0] <= F_UNIQUE and (feature_unique.astype(np.int64) == feature_unique).all():
        return True
    else:
        return False
