from sklearn.model_selection import StratifiedKFold, GroupKFold

from typing import Iterable


def cv_split_f(x, y, group_kf: Iterable = None, n_splits: int = 6) -> dict:
    """

    Args:
        x:
        y:
        group_kf:
        n_splits:

    Returns:

    """
    if group_kf is not None:
        gkf = GroupKFold(n_splits=n_splits)
        return dict(enumerate(gkf.split(X=x, y=y, groups=group_kf)))
    else:
        skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
        return dict(enumerate(skf.split(X=x, y=y)))
