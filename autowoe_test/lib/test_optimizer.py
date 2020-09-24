import sys
import warnings
import numpy as np

warnings.filterwarnings("ignore")
sys.path.append("..")

from autowoe.lib.optimizer import TreeParamOptimizer, StatOptimization, BinMerge, FMerge
from autowoe_test.conftest import data_setup, split_setup


# TODO: data_setup лучше заменить одним датасетом !!!


def test_tree_param_optimizer(data_setup, tree_setup):  # декартовое произведение параметров двух фикстур
   tree_param_optimizer = TreeParamOptimizer(data_setup, tree_setup)
   assert isinstance(tree_param_optimizer(3), dict)


def test_bin_merge_not_inf(data_setup):
    params = {"alpha": 0.1,
              "train_f": data_setup,
              "split": [0.0],
              "max_bin_count": None}

    b_merge = BinMerge(**params)
    split, is_drop = b_merge()

    assert not np.array_equal(split, np.array([-np.inf]))
    assert not is_drop

