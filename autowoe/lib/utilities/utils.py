"""Utility."""

from collections import namedtuple
from typing import Any
from typing import Callable
from typing import Dict
from typing import Hashable
from typing import Iterable
from typing import Literal
from typing import Set
from typing import Tuple
from typing import Union

import numpy as np

from strenum import StrEnum


Result = namedtuple("Result", ["score", "reg_alpha", "is_neg", "min_weights"])


class TaskType(StrEnum):
    """Solvable task types."""

    BIN: "TaskType" = "BIN"  # type: ignore
    REG: "TaskType" = "REG"  # type: ignore


def drop_keys(dict_: Dict, keys: Iterable[Hashable]) -> Dict:
    """Drop multiple keys from dict.

    Args:
        dict_: Dictonary.
        keys: Dropped keys.

    Returns:
        Filtered dictornary.

    """
    for key in keys:
        dict_.pop(key)
    return dict_


def flatten(d: dict, parent_key: str = "", sep: str = "_"):
    """Flatten dictonary of dictonaries.

    Args:
        d: Dictonary with nested dictonaries.
        parent_key: Parent outer key.
        sep: Separator for merged keys.

    Returns:
        Expanded dictonary.

    """
    items = []
    for k, v in d.items():
        new_key = parent_key + sep + k if parent_key else k
        if isinstance(v, dict):
            items.extend(flatten(v, new_key, sep=sep).items())
        else:
            items.append((new_key, v))
    return dict(items)


# Literal["BIN", "REG"]
def get_task_type(values: np.ndarray) -> TaskType:
    """Determine task type.

    Args:
        values: Array of values.

    Returns:
        task.

    """
    n_unique_values = np.unique(values).shape[0]

    task: Literal["BIN", "REG"]
    if n_unique_values == 1:
        raise RuntimeError("Only unique value in target")
    elif n_unique_values == 2:
        task = TaskType.BIN
    else:
        task = TaskType.REG

    return task


def feature_changing(
    feature_history: Dict[str, str],
    step_name: str,
    features_before: Union[Dict[str, str], Set[str]],
    func: Callable,
    *args,
    **kwargs
) -> Tuple[Any, Any]:
    """Safe feature filtering.

    Args:
        feature_history: History changes of features processing.
        step_name: Name of step.
        features_before: Features before processing.
        func: Filtering function.
        args: Function positional arguments.
        kwargs: Function named arguments.

    Returns:
        output:
        filter_features:

    """
    # features_before: Set[str]
    if isinstance(features_before, dict):
        features_before = set(features_before.keys())
    else:
        features_before = set(features_before)

    output, filter_features = func(*args, **kwargs)
    if isinstance(filter_features, dict):
        features_after = set(filter_features.keys())
    else:
        features_after = set(filter_features)

    features_diff = features_before - features_after
    for feature in features_diff:
        feature_history[feature] = step_name

    return output, filter_features
