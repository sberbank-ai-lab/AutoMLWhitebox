import sys
import warnings
import numpy as np

sys.path.append("..")
sys.path.append("../..")
warnings.filterwarnings("ignore")

from autowoe.lib.pipelines.pipeline_homotopy import HTransform


def test_tree_param_optimizer_type(data_setup, tree_params):
    htransform = HTransform(data_setup["feature"],
                            data_setup["target"],
                            tree_params)

    assert isinstance(htransform(0), np.ndarray)


# def test_tree_param_optimizer_values(data_setup, tree_params):
#     htransform = HTransform(data_setup["feature"],
#                             data_setup["target"],
#                             tree_params)

#     assert np.array_equal(htransform(0), data_setup[1])



