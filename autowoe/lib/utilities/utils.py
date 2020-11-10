from collections import namedtuple
from typing import Dict, Iterable, Hashable

Result = namedtuple("Result", ["score", "reg_alpha", "is_neg", "min_weights"])


def drop_keys(dict_: Dict, keys: Iterable[Hashable]) -> Dict:
    """
    Drop multiple keys from dict

    Args:
        dict_:
        keys:

    Returns:

    """
    for key in keys:
        dict_.pop(key)
    return dict_


def flatten(d: dict, parent_key: str = '', sep: str = '_'):
    """

    Args:
        d:
        parent_key:
        sep:

    Returns:

    """
    items = []
    for k, v in d.items():
        new_key = parent_key + sep + k if parent_key else k
        if isinstance(v, dict):
            items.extend(flatten(v, new_key, sep=sep).items())
        else:
            items.append((new_key, v))
    return dict(items)
