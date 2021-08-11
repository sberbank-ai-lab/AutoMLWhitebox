"""Cross validation utilities."""

from typing import Iterable
from typing import Optional

import numpy as np

from sklearn.model_selection import GroupKFold
from sklearn.model_selection import StratifiedKFold

from autowoe.lib.utilities.utils import TaskType


def cv_split_f(x, y, task: TaskType, group_kf: Iterable = None, n_splits: int = 6, random_state: int = 42) -> dict:
    """Get CV-splits.

    Args:
        x: Features.
        y: Target.
        task: Task.
        group_kf: Groups.
        n_splits: Number of splits.
        random_state: Random state.

    Returns:
        CV-splits.

    """
    if task == TaskType.BIN:
        if group_kf is not None:
            gkf = GroupKFold(n_splits=n_splits)
            return dict(enumerate(gkf.split(X=x, y=y, groups=group_kf)))
        else:
            skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
            return dict(enumerate(skf.split(X=x, y=y)))
    else:
        skf = StratifiedKFoldReg(n_splits=n_splits, shuffle=True, random_state=random_state)
        return dict(enumerate(skf.split(X=x, y=y)))


class StratifiedKFoldReg(StratifiedKFold):
    """Stratification for continuous variable.

    Stratification method 'sorted' was taken from:
        (https://github.com/scikit-learn/scikit-learn/issues/4757)

    Args:
        method: Method for stratification
        n_y_bins: Number of target bins. Default: None.

    """

    def __init__(self, method: Optional[str] = None, n_y_bins: Optional[int] = None, **kwargs):
        self._method = method
        self._n_y_bins = n_y_bins

        super().__init__(**kwargs)

    def split(self, X, y, groups=None):
        """Generate indices to split data into training and test set."""
        if self._method is None:
            return self._sorted_split(X, y, groups)
        else:
            raise NotImplementedError

    def _sorted_split(self, X, y, groups=None):
        n_samples = len(y)

        n_labels = int(np.floor(n_samples / self.n_splits))
        y_labels_sorted = np.concatenate([np.repeat(ii, self.n_splits) for ii in range(n_labels)])

        mod = np.mod(n_samples, self.n_splits)

        _, labels_idx = np.unique(y_labels_sorted, return_index=True)
        rand_label_ix = np.random.choice(labels_idx, mod, replace=False)
        y_labels_sorted = np.insert(y_labels_sorted, rand_label_ix, y_labels_sorted[rand_label_ix])

        map_labels_y = dict()
        for ix, label in zip(np.argsort(y), y_labels_sorted):
            map_labels_y[ix] = label

        y_labels = np.array([map_labels_y[ii] for ii in range(n_samples)])

        return super().split(X, y_labels, groups)

    def _bins_split(self, X, y, groups=None):
        y_labels = y
        return super().split(X, y_labels, groups)
