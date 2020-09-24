import pytest
import pandas as pd
import numpy as np

from collections import OrderedDict


@pytest.fixture(scope="module",
                params=["d1.csv", "d2.csv", "d3.csv", "d4.csv", "d5.csv"],
                ids=["d1.csv", "d2.csv", "d3.csv", "d4.csv", "d5.csv"])
def data_setup(request):
    data = pd.read_csv("./resources/" + request.param)
    yield data
    del data


@pytest.fixture(scope="module", params=[
                [-10, 1, 100, 200, 500],
                [0],
                [-np.inf],
                [np.inf],
                [-np.inf, np.inf],
                [-np.inf, 100, 500],
                [0, 100, np.inf],
                [-10, 10],
                np.arange(-100, 500, 2)],
                ids=["ordinary", "zero", "-inf", "inf", "two_inf", "ordinary_-inf",
                     "ordinary_inf", "sym_ordinary", "large_ordinary"])
def split_setup(request):
    split = request.param
    yield split
    del split


@pytest.fixture(scope="function",
                params=[OrderedDict({"max_depth": (3, 4),
                                     "min_child_samples": (100, 200, 400),
                                     "min_gain_to_split": (0, 0.05, 0.1)}),
                        OrderedDict({"min_child_samples": (100, 200, 400),
                                     "min_gain_to_split": (0, 0.05, 0.1)}),
                        OrderedDict({"max_depth": (3, 4, 5),
                                     "min_gain_to_split": (0, 0.05, 0.1)})],
                ids=["0", "1", "2"])
def tree_setup(request):
    split = request.param
    yield split
    del split


@pytest.fixture(scope="function",
                params=[{"max_depth": 4,
                         "min_child_samples": 200,
                         "min_gain_to_split": 0.05},
                        {"max_depth": 5,
                         "min_child_samples": 500,
                         "min_gain_to_split": 0},
                        {"max_depth": 3,
                         "min_child_samples": 50,
                         "min_gain_to_split": 0.1}],
                ids=["0", "1", "2"])
def tree_params(request):
    split = request.param
    yield split
    del split

# TODO: Add Bin test !!!






