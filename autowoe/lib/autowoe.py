"""AutoWoe."""

import collections

from collections import OrderedDict
from copy import deepcopy
from multiprocessing import Pool
from typing import Any
from typing import Dict
from typing import Hashable
from typing import List
from typing import Optional
from typing import Sequence
from typing import Tuple
from typing import Union

import numpy as np
import pandas as pd

# from pandas.core.arrays.base import try_cast_to_ea
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler

from .cat_encoding.cat_encoding import CatEncoding
from .logging import get_logger
from .logging import verbosity_to_loglevel
from .optimizer.optimizer import TreeParamOptimizer
from .pipelines.pipeline_feature_special_values import CATEGORY_SPECIAL_SET
from .pipelines.pipeline_feature_special_values import DEFAULT_OPTIONS_SPECIAL_VALUES
from .pipelines.pipeline_feature_special_values import EXTEND_OPTIONS_SPECIAL_VALUES
from .pipelines.pipeline_feature_special_values import REAL_SPECIAL_SET
from .pipelines.pipeline_feature_special_values import FeatureSpecialValues
from .pipelines.pipeline_homotopy import HTransform
from .selectors.selector_first import feature_imp_selector
from .selectors.selector_first import nan_constant_selector
from .selectors.selector_last import Selector
from .types_handler.types_handler import TypesHandler
from .utilities.cv_split_f import cv_split_f
from .utilities.refit import refit_reg
from .utilities.refit import refit_simple
from .utilities.sql import get_sql_inference_query
from .utilities.utils import TaskType
from .utilities.utils import feature_changing
from .utilities.utils import get_task_type
from .woe.woe import WoE


logger = get_logger(__name__)

SplitType = Optional[Union[np.ndarray, List[float], Dict[int, int]]]


def get_monotonic_constr(task: TaskType, name: str, train: pd.DataFrame, target: str, spec_values: Optional[List[Any]]):
    """Check monotonic constraint."""
    df = train[[target, name]].dropna()
    if spec_values is not None:
        df = df.loc[~df[name].isin(spec_values)]
    try:
        if task == TaskType.BIN:
            auc = roc_auc_score(df[target].values, df[name].values)
            return str(int(np.sign(auc - 0.5)))
        elif task == TaskType.REG:
            corr = df[[target, name]].corr().iloc[0, 1]
            return str(int(np.sign(corr)))

    except (ValueError, TypeError, IndexError):
        return "0"


class AutoWoE:
    """Implementation of Logistic regression with WoE transformation.

    Args:
        task: TaskType
            Type of task: 'TaskType.BIN' or 'TaskType.REG'.
        interpreted_model: bool
            Model interpretability flag.
        monotonic: bool
            Global condition for monotonic constraints. If "True", then only
            monotonic binnings will be built. You can pass values to the .fit
            method that change this condition separately for each feature.
        max_bin_count: int
            Global limit for the number of bins. Can be specified for every
            feature in .fit
        select_type: None or int
            The type to specify the primary feature selection. If the type is an integer,
            then we select the number of features indicated by this number (with the best feature_importance).
            If the value is "None", we leave only features with feature_importance greater than 0.
        pearson_th:  0 < pearson_th < 1
            Threshold for feature selection by correlation. All features with
            the absolute value of correlation coefficient greater then
            pearson_th will be discarded.
        metric_th: Threshold for feature selection by one-dimensional metric.
            WoE will be discarded when:
                1) Binary task: AUC < metric_th
                    Threshold: 0.5 < metric_th < 1
                2) Regression task: R2 < metric_th
                    Threshold: 0.0 < metric_th < 1
        vif_th: vif_th > 0
            Threshold for feature selection by VIF. Features with VIF > vif_th
            are iteratively discarded one by one, then VIF is recalculated
            until all VIFs are less than vif_th.
        imp_th: real >= 0
            Threshold for feature selection by feature importance
        th_const:
            Threshold, which determines that the feature is constant.
            If the number of valid values is greater than the threshold, then
            the column is not constant. For float, the number of
            valid values will be calculated as the sample size * th_const
        force_single_split: bool
            In the tree parameters, you can set the minimum number of
            observations in the leaf. Thus, for some features, splitting for 2 beans at least will be impossible. If you specify that
            force_single_split = True, it means that 1 split will be created for the feature, if the minimum bin size is greater than th_const.
        th_nan: int >= 0
            Threshold, which determines that WoE values are calculated to NaN.
        th_cat: int >= 0
            Threshold, which determines which categories are small.
        th_mark: int >= 0
            Threshold, which determines which group of marked values are small.
        woe_diff_th: float = 0.01
            The option to merge NaNs and rare categories with another bin,
            if the difference in WoE is less than woe_diff_th
        min_bin_size: int > 1, 0 < float < 1
            Minimum bin size when splitting.
        min_bin_mults: list of floats > 1
            If minimum bin size is specified, you can specify a list to check
            if large values work better, for example: [2, 4]
        min_gains_to_split: list of floats >= 0
            min_gain_to_split values that will be iterated to find the best split.
        metric_tol: 1e-5 <= metric_tol <=1e-2
            Metric tolerance. You can lower the metric_tol value from the maximum
            to make the model simpler {Binary -> AUC, Regression -> R2}.
        cat_alpha: float > 0
            Regularizer for category encoding.
        cat_merge_to: str
            The way of WoE values filling in the test sample for categories
            that are not in the training sample.
            Values - 'to_nan', 'to_woe_0', 'to_maxfreq', 'to_maxp', 'to_minp'
        nan_merge_to: str
            The way of WoE values filling on the test sample for real NaNs,
            if they are not included in their group.
            Values - 'to_woe_0', 'to_maxfreq', 'to_maxp', 'to_minp'
        mark_merge_to: str
            The way of WoE values filling on the test sample for 'markes`,
            if they are not included in their group.
            Values - 'to_woe_0', 'to_maxfreq', 'to_maxp', 'to_minp'
        oof_woe: bool
            Use OOF or standard encoding for WOE.
        n_folds: int
            Number of folds for feature selection / encoding, etc.
        n_jobs: int > 0
            Number of CPU cores to run in parallel.
        l1_grid_size: real > 0
            Grid size in l1 regularization
        l1_exp_scale: real > 1
            Grid scale in l1 regularization
        imp_type: str
            Feature importances type. Feature_imp and perm_imp are available.
            It is used to sort the features at the first and at the final
            stage of feature selection.
        regularized_refit: bool
            Use regularization at the time of model refit. Otherwise, we have
            a statistical model.
        p_val: 0 < p_val <= 1
            When training a statistical model, do backward selection
            until all p-values of the model's coefficient are less than p_val
        verbose: int >= 0
            verbosity level
        debug: bool
            Debug mode
        **kwargs: Deprecated parameters.

    """

    @property
    def weights(self):
        """Weights."""
        return self._weights

    @property
    def intercept(self):
        """Intercet."""
        return self._intercept

    @property
    def p_vals(self):
        """P-values."""
        return self._p_vals

    # TODO: Merge params for BIN and REG tasks
    def __init__(
        self,
        task: Optional[TaskType] = "BIN",
        interpreted_model: bool = True,
        monotonic: bool = False,
        max_bin_count: int = 5,
        select_type: Optional[int] = None,
        pearson_th: float = 0.9,
        metric_th: Optional[float] = None,
        vif_th: float = 5.0,
        imp_th: float = 0.001,
        th_const: Union[int, float] = 0.005,
        force_single_split: bool = False,
        th_nan: Union[int, float] = 0.005,
        th_cat: Union[int, float] = 0.005,
        th_mark: Union[int, float] = 0.005,
        woe_diff_th: float = 0.01,
        min_bin_size: Union[int, float] = 0.01,
        min_bin_mults: Sequence[float] = (2, 4),
        min_gains_to_split: Sequence[float] = (0.0, 0.5, 1.0),
        metric_tol: float = 1e-4,
        cat_alpha: float = 1,
        cat_merge_to: str = "to_woe_0",
        nan_merge_to: str = "to_woe_0",
        mark_merge_to: str = "to_woe_0",
        oof_woe: bool = False,
        n_folds: int = 6,
        n_jobs: int = 10,
        l1_grid_size: int = 20,
        l1_exp_scale: float = 4,
        imp_type: str = "feature_imp",
        regularized_refit: bool = True,
        p_val: float = 0.05,
        debug: bool = False,
        verbose: int = 2,
        **kwargs,
    ):
        logger.setLevel(verbosity_to_loglevel(verbose))

        assert (
            nan_merge_to in DEFAULT_OPTIONS_SPECIAL_VALUES
        ), "Value for nan_merge_to is invalid. Valid are [{}]".format(DEFAULT_OPTIONS_SPECIAL_VALUES)

        assert (
            cat_merge_to in EXTEND_OPTIONS_SPECIAL_VALUES
        ), "Value for cat_merge_to is invalid. Valid are [{}]".format(EXTEND_OPTIONS_SPECIAL_VALUES)

        assert (
            mark_merge_to in EXTEND_OPTIONS_SPECIAL_VALUES
        ), "Value for mari_merge_to is invalid. Valid are [{}]".format(EXTEND_OPTIONS_SPECIAL_VALUES)

        self._params = {
            "task": task,
            "interpreted_model": interpreted_model,
            "monotonic": monotonic,
            "max_bin_count": max_bin_count,
            "select_type": select_type,
            "pearson_th": pearson_th,
            "metric_th": metric_th,
            "vif_th": vif_th,
            "imp_th": imp_th,
            "min_bin_mults": min_bin_mults,
            "min_gains_to_split": min_gains_to_split,
            "force_single_split": force_single_split,
            "metric_tol": metric_tol,
            "cat_alpha": cat_alpha,
            "cat_merge_to": cat_merge_to,
            "nan_merge_to": nan_merge_to,
            "mark_merge_to": mark_merge_to,
            "oof_woe": oof_woe,
            "n_folds": n_folds,
            "n_jobs": n_jobs,
            "l1_grid_size": l1_grid_size,
            "l1_exp_scale": l1_exp_scale,
            "imp_type": imp_type,
            "population_size": None,
            "regularized_refit": regularized_refit,
            "p_val": p_val,
            "debug": debug,
            "th_const": th_const,
            "th_nan": th_nan,
            "th_cat": th_cat,
            "th_mark": th_mark,
            "woe_diff_th": woe_diff_th,
            "min_bin_size": min_bin_size,
        }
        for deprecated_arg, new_arg in zip(
            ["l1_base_step", "l1_exp_step", "population_size", "feature_groups_count", "auc_th", "auc_tol"],
            ["l1_grid_size", "l1_exp_scale", None, None, "metric_th", "metric_tol"],
        ):
            if deprecated_arg in kwargs:
                msg = "Parameter {0} is deprecated.".format(deprecated_arg)
                if new_arg is not None:
                    msg = msg + " Value will be set to {0} parameter, but exception will be raised in future.".format(
                        new_arg
                    )
                    self._params[new_arg] = kwargs[deprecated_arg]
                logger.warning(msg, DeprecationWarning)

        self.woe_dict = None
        self.train_df = None
        self.split_dict = None  # словарь со сплитами для каждого признкака
        self.target = None  # целевая переменная
        self.clf = None  # модель лог регрессии
        self.features_fit = None  # Признаки, которые прошли проверку Selector + информация о лучшей итерации Result
        self._cv_split = None  # Словарь с индексами разбиения на train и test
        self._small_nans = None

        self._private_features_type = None
        self._public_features_type = None

        self._weights = None
        self._intercept = None
        self._p_vals = None
        self._threshold = 0.0

        self.feature_history = None

    @property
    def features_type(self):  # noqa: D102
        return self._public_features_type

    @property
    def private_features_type(self):  # noqa: D102
        return self._private_features_type

    def get_split(self, feature: Hashable):  # noqa: D102
        return self.woe_dict[feature].split

    def get_woe(self, feature_name: Hashable):
        """Get WoE for feature."""
        if self.private_features_type[feature_name] == "real":
            split = self.woe_dict[feature_name].split.copy()
            woe = self.woe_dict[feature_name].cod_dict
            split = enumerate(np.hstack([split, [np.inf]]))
            split = OrderedDict(split)

            spec_val = set(woe.keys()) - set(split.keys())
            spec_val = OrderedDict((key, woe[key]) for key in spec_val)

            split = OrderedDict((split[key], value) for (key, value) in woe.items() if key in split)
            split, spec_val = list(split.items()), list(spec_val.items())

            borders, values = list(zip(*split))
            new_borders = list(zip([-np.inf] + list(borders[:-1]), borders))
            new_borders = [("{:.2f}".format(x[0]), "{:.2f}".format(x[1])) for x in new_borders]

            split = list(zip(new_borders, values)) + spec_val

        elif self.private_features_type[feature_name] == "cat":
            split = list(self.woe_dict[feature_name].cod_dict.items())
        else:
            raise ValueError(f"Feature type {self.private_features_type[feature_name]} is not supported")

        split = [(x[1], str(x[0])) for x in split]
        return pd.Series(*(zip(*split)))

    def _infer_params(
        self,
        train: pd.DataFrame,
        target_name: str,
        features_type: Optional[Dict[str, str]] = None,
        group_kf: Hashable = None,
        max_bin_count: Optional[Dict[str, int]] = None,
        features_monotone_constraints: Optional[Dict[str, str]] = None,
        features_mark_values: Optional[Dict[str, Tuple[Any]]] = None,
    ):
        self.params = deepcopy(self._params)

        if self.params["task"] is None:
            task = get_task_type(train[target_name].values)
            self.params["task"] = task

        for k in ["th_const", "th_nan", "th_cat", "min_bin_size"]:
            val = self.params[k]
            self.params[k] = int(val * train.shape[0]) if 0 <= val < 1 else int(val)

        min_data_in_bin = [
            self.params["min_bin_size"],
        ]
        for m in self.params["min_bin_mults"]:
            min_data_in_bin.append(int(m * self.params["min_bin_size"]))

        self._tree_dict_opt = OrderedDict(
            {
                "min_data_in_leaf": (self.params["min_bin_size"],),
                "min_data_in_bin": min_data_in_bin,
                "min_gain_to_split": self.params["min_gains_to_split"],
            }
        )

        self._features_mark_values = features_mark_values

        # составим features_type
        self._features_type = features_type
        if self._features_type is None:
            self._features_type = {}

        assert target_name not in self._features_type, "target_name in features_type!!!"
        assert group_kf not in self._features_type, "group_kf in features_type!!!"

        droplist = [target_name]
        if group_kf is not None:
            droplist.append(group_kf)

        for col in train.columns.drop(droplist):
            if col not in self._features_type:
                self._features_type[col] = None

        # поработаем с монотонными ограничениями
        self.features_monotone_constraints = features_monotone_constraints
        if self.features_monotone_constraints is None:
            self.features_monotone_constraints = {}

        checklist = ["auto"]
        if self.params["monotonic"]:
            checklist.extend(["0", 0, None])

        for col in self._features_type:
            val = self.features_monotone_constraints.get(col)

            if val in checklist:
                new_val = get_monotonic_constr(self.params["task"], col, train, target_name, features_mark_values)
            elif val in ["0", 0, None]:
                new_val = "0"
            else:
                new_val = val

            self.features_monotone_constraints[col] = new_val

        # max_bin_count
        self.max_bin_count = max_bin_count
        if self.max_bin_count is None:
            self.max_bin_count = {}

        for col in self._features_type:
            if col not in self.max_bin_count:
                self.max_bin_count[col] = self.params["max_bin_count"]

    def fit(
        self,
        train: pd.DataFrame,
        target_name: str,
        features_type: Optional[Dict[str, str]] = None,
        group_kf: Hashable = None,
        max_bin_count: Optional[Dict[str, int]] = None,
        features_monotone_constraints: Optional[Dict[str, str]] = None,
        features_mark_values: Optional[Dict[str, Tuple[Any]]] = None,
        validation: Optional[pd.DataFrame] = None,
    ):
        """Train model.

        Args:
            train: Training sample.
            target_name: Target variable's column name
            features_type: Dictionary with feature types, "cat" - categorical, "real" - real, "date" - for date
            group_kf: Column name for GroupKFold
            max_bin_count: Dictionary with feature name -> maximum bin quantity values
            features_monotone_constraints: Dictionary with monotonic constraints for features.
                "-1" - the feature values decreases monotonically when the target variable's value increases
                "0" - no limitations. Switches to auto in case of monotonic = True
                "1" - the feature values monotonically increases when the target variable's value increases
                "auto" - the feature values monotonically changes.
                Not specified for categorical features.
            features_mark_values: Marked values of feature which will be processed like `NaN` or small categories.
            validation: Additional validation sample used for model selection. Currently supported:
                - feature selection by p-value

        """
        self._infer_params(
            train,
            target_name,
            features_type,
            group_kf,
            max_bin_count,
            features_monotone_constraints,
            features_mark_values,
        )

        if group_kf:
            group_kf = train[group_kf].values

        types_handler = TypesHandler(
            train=train,
            public_features_type=self._features_type,
            max_bin_count=self.max_bin_count,
            features_monotone_constraints=self.features_monotone_constraints,
        )

        (
            train_,
            self._public_features_type,
            self._private_features_type,
            max_bin_count,
            features_monotone_constraints,
        ) = types_handler.transform()
        del types_handler

        train_ = train_[[*self.private_features_type.keys(), target_name]]
        self.target = train_[target_name]
        self.feature_history = {key: None for key in self.private_features_type.keys()}

        # Remove columns with huge ratio of NaN-values
        train_, self._private_features_type = feature_changing(
            self.feature_history,
            "NaN values",
            self._private_features_type,
            nan_constant_selector,
            train_,
            self.private_features_type,
            th_const=self.params["th_const"],
        )

        # Target preprocessing
        self._preprocess_target()

        # Remove featuters by model importance
        train_, self._private_features_type = feature_changing(
            self.feature_history,
            "Low importance",
            self.private_features_type,
            feature_imp_selector,
            train_,
            self.params["task"],
            self.private_features_type,
            target_name,
            imp_th=self.params["imp_th"],
            imp_type=self.params["imp_type"],
            select_type=self.params["select_type"],
            process_num=self.params["n_jobs"],
        )

        # Fill small group of category features, NaN-values by tags
        self._features_special_values = FeatureSpecialValues(
            th_nan=self.params["th_nan"],
            th_cat=self.params["th_cat"],
            cat_merge_to=self.params["cat_merge_to"],
            nan_merge_to=self.params["nan_merge_to"],
            mark_merge_to=self.params["mark_merge_to"],
            marked_values=self._features_mark_values,
        )
        train_, spec_values = self._features_special_values.fit_transform(
            train=train_, features_type=self.private_features_type
        )

        self._cv_split = cv_split_f(train_, self.target, self.params["task"], group_kf, n_splits=self.params["n_folds"])

        params_gen = (
            (
                x,
                deepcopy(train_[[x, target_name]]),
                self.params["task"],
                features_monotone_constraints[x],
                max_bin_count[x],
                self.params["cat_alpha"],
            )
            for x in self.private_features_type.keys()
        )

        if self.params["n_jobs"] > 1:
            with Pool(self.params["n_jobs"]) as pool:
                result = pool.starmap(self.feature_woe_transform, params_gen)
        else:
            result = []
            for params in params_gen:
                result.append(self.feature_woe_transform(*params))

        split_dict = dict(zip(self.private_features_type.keys(), result))
        split_dict = {key: split_dict[key] for key in split_dict if split_dict[key] is not None}

        _, self._private_features_type = feature_changing(
            self.feature_history,
            "Unable to WOE transform",
            self._private_features_type,
            lambda d: (None, {x: self.private_features_type[x] for x in d if x in d.keys()}),
            split_dict,
        )

        logger.info(f"{split_dict.keys()} to selector !!!!!")
        self.split_dict = split_dict  # набор пар признаки - границы бинов
        self.train_df = self._train_encoding(train_, spec_values, self.params["oof_woe"])

        logger.info("Feature selection...")
        selector = Selector(
            interpreted_model=self.params["interpreted_model"],
            task=self.params["task"],
            train=self.train_df,
            target=self.target,
            features_type=self.private_features_type,
            n_jobs=self.params["n_jobs"],
            cv_split=self._cv_split,
        )

        best_features, self._sel_result = selector(
            self.feature_history,
            pearson_th=self.params["pearson_th"],
            metric_th=self.params["metric_th"],
            vif_th=self.params["vif_th"],
            l1_grid_size=self.params["l1_grid_size"],
            l1_exp_scale=self.params["l1_exp_scale"],
            metric_tol=self.params["metric_tol"],
        )

        # create validation data if it's defined and usefull
        valid_enc, valid_target = None, None
        if validation is not None and not self.params["regularized_refit"]:
            valid_enc = self.test_encoding(validation, best_features)
            valid_target = validation[target_name]

        fit_result, _ = feature_changing(
            self.feature_history,
            "Pruned during regression refit",
            self._private_features_type,
            self._model_fit,
            self.train_df,
            best_features,
            valid_enc,
            valid_target,
        )

        for p in [("features_fit", False), ("weights", True), ("intercept", True), ("b_vars", True), ("p_vals", True)]:
            nm, is_private = p
            attr_name = "{}{}".format("_" * is_private, nm)
            if nm in fit_result:
                setattr(self, attr_name, fit_result[nm])

        if not self.params["debug"]:
            del self.train_df
            del self.target

    def feature_woe_transform(
        self,
        feature_name: str,
        train_df: pd.DataFrame,
        task: TaskType,
        features_monotone_constraints: str,
        max_bin_count: int,
        cat_alpha: float = 1.0,
    ) -> SplitType:
        """Transformation WoE.

        Args:
            feature_name: Feature column name.
            train_df: Train dataset.
            task: Task.
            features_monotone_constraints: Feature mononotic contr.
            max_bin_count: Maximum bin counts.
            cat_alpha: Alpha.

        Returns:
            Transformed feature.

        """
        train_df = train_df.reset_index(drop=True)
        logger.info(f"{feature_name} processing...")
        target_name = train_df.columns[1]
        # Откидываем здесь закодированные маленькие категории/наны. Их не учитываем при определения бинов
        if np.issubdtype(train_df.dtypes[feature_name], np.number):
            nan_index = []
        else:
            sn_set = CATEGORY_SPECIAL_SET if self.private_features_type[feature_name] == "cat" else REAL_SPECIAL_SET
            nan_index = train_df[feature_name].isin(sn_set)
            nan_index = np.where(nan_index.values)[0]

        cat_enc = None
        if self.private_features_type[feature_name] == "cat":
            cat_enc = CatEncoding(data=train_df)
            train_df = cat_enc(self._cv_split, nan_index, cat_alpha)

        train_df = train_df.iloc[np.setdiff1d(np.arange(train_df.shape[0]), nan_index), :]

        if self.params["task"] == TaskType.BIN:
            train_df = train_df.astype({feature_name: float, target_name: int})
        else:
            train_df = train_df.astype({feature_name: float, target_name: float})
        # нужный тип для lgb после нанов и маленьких категорий
        if train_df.shape[0] == 0:  # случай, если кроме нанов и маленьких категорий ничего не осталось
            split = [-np.inf]
            if self.private_features_type[feature_name] == "cat":
                return cat_enc.mean_target_reverse(split)
            elif self.private_features_type[feature_name] == "real":
                return split
            else:
                raise ValueError("self.features_type[feature] is cat or real")

        # подбор оптимальных параметров дерева
        tree_dict_opt = deepcopy(self._tree_dict_opt)
        if max_bin_count:  # ограничение на число бинов

            leaves_range = tuple(range(2, max_bin_count + 1))
            tree_dict_opt = OrderedDict(
                {
                    **self._tree_dict_opt,
                    **{"num_leaves": leaves_range, "bin_construct_sample_cnt": (int(1e8),)},
                }
            )

            # Еще фича force_single_split ..
            if self.params["force_single_split"]:
                min_size = train_df.shape[0] - train_df[feature_name].value_counts(dropna=False).values[0]
                if self.params["th_const"] < min_size < self.params["min_bin_size"]:
                    tree_dict_opt["min_data_in_leaf"] = [min_size]
                    tree_dict_opt["min_data_in_bin"] = [3]
                    tree_dict_opt["num_leaves"] = [2]

        tree_opt = TreeParamOptimizer(
            data=train_df,
            task=task,
            n_folds=self.params["n_folds"],
            params_range=collections.OrderedDict(
                **tree_dict_opt, **{"monotone_constraints": (features_monotone_constraints,)}
            ),
        )
        tree_param = tree_opt(3)
        # значение monotone_constraints содержится в tree_params
        # подбор подходяшего сплита на бины
        htransform = HTransform(self._params["task"], train_df[feature_name], train_df[target_name])
        split = htransform(tree_param)

        #  Обратная операция к mean_target_encoding
        if self.private_features_type[feature_name] == "cat":
            return cat_enc.mean_target_reverse(split)
        elif self.private_features_type[feature_name] == "real":
            return split
        else:
            raise ValueError("self.features_type[feature] is cat or real")

    def _get_task_type(values: np.ndarray) -> TaskType:
        n_unique_values = np.unique(values).shape[0]
        if n_unique_values == 1:
            raise RuntimeError("Only unique value in target")
        elif n_unique_values == 2:
            task = TaskType.BIN
        else:
            task = TaskType.REG

        return task

    def _preprocess_target(self):
        if self.params["task"] == TaskType.REG:
            self._target_scaler = StandardScaler()
            target_values = self._target_scaler.fit_transform(self.target.values.reshape(-1, 1))
            self.target.loc[:] = target_values.ravel()
            self._target_std = self._target_scaler.mean_[0]
            self._target_mean = self._target_scaler.scale_[0]

    def _train_encoding(self, train: pd.DataFrame, spec_values: Dict, folds_codding: bool) -> pd.DataFrame:  # TODO: ref
        """Encode a train dataset based on WoE estimates."""
        woe_dict = dict()
        woe_list = []
        for feature in self.private_features_type:
            woe = WoE(
                f_type=self.private_features_type[feature],
                split=self.split_dict[feature],
                woe_diff_th=self.params["woe_diff_th"],
                target_type=self.params["task"],
            )
            if folds_codding:
                df_cod = woe.fit_transform_cv(
                    train[feature], self.target, spec_values=spec_values[feature], cv_index_split=self._cv_split
                )
                woe.fit(train[feature], self.target, spec_values=spec_values[feature])
            else:
                df_cod = woe.fit_transform(train[feature], self.target, spec_values=spec_values[feature])
            woe_dict[feature] = woe
            woe_list.append(df_cod)
        self.woe_dict = woe_dict
        train_tr = pd.concat(woe_list, axis=1)
        train_tr.columns = self.private_features_type.keys()
        return train_tr

    def _model_fit(self, data_enc, features, valid_enc=None, valid_target=None) -> dict:
        """Final model training."""
        x_train, y_train = data_enc[features].values, self.target.values
        x_val, y_val = None, None
        p_vals = None

        result = dict()
        if self.params["regularized_refit"]:
            w, i, interp_feat_flag = refit_reg(
                self.params["task"],
                x_train,
                y_train,
                l1_grid_size=self.params["l1_grid_size"],
                l1_exp_scale=self.params["l1_exp_scale"],
                max_penalty=self._sel_result.reg_alpha,
                interp=self.params["interpreted_model"],
            )
        else:
            if valid_enc is not None:
                x_val, y_val = valid_enc[features].values, valid_target.values

            w, i, interp_feat_flag, p_vals, b_var = refit_simple(
                self.params["task"],
                x_train,
                y_train,
                interp=self.params["interpreted_model"],
                p_val=self.params["p_val"],
                x_val=x_val,
                y_val=y_val,
            )

            if b_var is not None:
                result["b_var"] = b_var

        _feats = np.array(features)[interp_feat_flag]

        # features_before = set(features)
        features_fit = pd.Series(w, _feats)
        result["features_fit"] = features_fit
        # features_after = set(features_fit.index)
        # features_diff = features_before - features_after
        # if feature_history is not None:
        #     for feature in features_diff:
        #         feature_history[feature] = 'Pruned during regression refit'

        if not self.params["regularized_refit"]:
            result["p_vals"] = pd.Series(p_vals, list(_feats) + ["Intercept_"])

        logger.info(features_fit)
        result["weights"] = w
        result["intercept"] = i

        return result, features_fit

    def test_encoding(self, test: pd.DataFrame, feats: Optional[List[str]] = None) -> pd.DataFrame:
        """Encode a test dataset based on WoE estimates.

        Args:
            test: Test dataset.
            feats: List of features.

        Returns:
            Features encoding.

        """
        if feats is None:
            feats = list(self.features_fit.index)

        feats_to_get = deepcopy(feats)

        for feat in feats:
            parts = feat.split("__F__")
            if len(parts) > 1:
                feats_to_get.append("__F__".join(parts[:-1]))
        feats_to_get = [x for x in list(set(feats_to_get)) if x in test.columns]

        types = {}
        for feat in feats_to_get:
            if feat in self._public_features_type:
                types[feat] = self._public_features_type[feat]

        types_handler = TypesHandler(train=test[feats_to_get], public_features_type=types)
        test_, _, _, _, _ = types_handler.transform()
        del types_handler

        woe_list = []
        test_, spec_values = self._features_special_values.transform(test_, feats)
        # здесь дебажный принт
        logger.debug(spec_values)
        for feature in feats:
            df_cod = self.woe_dict[feature].transform(test_[feature], spec_values[feature])
            woe_list.append(df_cod)

        test_tr = pd.concat(woe_list, axis=1)
        test_tr.columns = feats
        return test_tr[feats]

    def predict(self, test: pd.DataFrame) -> np.ndarray:
        """Make prediction for a test dataset.

        Available for binary and regression tasks

        Args:
            test: pd.DataFrame

        Returns:
            np.ndarray

        """
        lin_pred = self._predict(test)

        if self.params["task"] == TaskType.BIN:
            prediction = (lin_pred >= self._threshold).astype(np.int32)
        else:
            prediction = self._target_scaler.inverse_transform(lin_pred)

        return prediction

    def predict_proba(self, test: pd.DataFrame) -> np.ndarray:
        """Make predictions for a test dataset (only for binary task!).

        Args:
            test: pd.DataFrame

        Returns:
            np.ndarray

        """
        assert self.params["task"] == TaskType.BIN, "Method 'predict_proba' is available only for binary task"

        lin_pred = self._predict(test)
        prob = 1 / (1 + np.exp(-lin_pred))

        return prob

    def _predict(self, test: pd.DataFrame):
        test_tr = self.test_encoding(test)
        lin_pred = np.dot(test_tr.values, self.weights) + self.intercept

        return lin_pred

    def get_model_represenation(self):
        """Get scorecard.

        Returns:
            scorecard.

        """
        features = list(self.features_fit.index)
        result = dict()
        for feature in features:
            feature_data = dict()
            woe = self.woe_dict[feature]
            feature_data["f_type"] = woe.f_type

            if woe.f_type == "real":
                feature_data["splits"] = [0 + round(float(x), 6) for x in woe.split]
            else:
                feature_data["cat_map"] = {str(k): int(v) for k, v in woe.split.items()}
                spec_vals = self._small_nans.cat_encoding[feature]
                feature_data["spec_cat"] = (spec_vals[0], spec_vals[2])

            feature_data["cod_dict"] = {
                int(k): (0 + round(float(v), 6)) for k, v in woe.cod_dict.items() if type(k) is int or type(k) is float
            }

            feature_data["weight"] = float(self.features_fit[feature])
            feature_data["nan_value"] = self._small_nans.all_encoding[feature]
            feature_data["spec_cod"] = {k: (0 + round(float(v), 6)) for k, v in woe.cod_dict.items() if type(k) is str}

            result[feature] = feature_data

        return {"features": result, "intercept": float(self.intercept)}

    def get_sql_inference_query(
        self,
        table_name,
        round_digits=3,
        round_features=5,
        output_name="PROB",
        alias="WOE_TAB",
        bypass_encoded=True,
        template=None,
        nan_pattern_numbers="({0} IS NULL OR {0} = 'NaN')",
        nan_pattern_category="({0} IS NULL OR LOWER(CAST({0} AS VARCHAR(50))) = 'nan')",
        preprocessing=None,
    ) -> str:
        """Get inference query for whitebox model.

        Args:
            table_name: Source table name that should be passed into query
            round_digits: round woe and coefs to simplify query. Note: may be little accuracy decrease
            round_features: round features to simplify query. Note: may be little accuracy decrease
            output_name: name of output prediction feature
            alias: alias of woe_table in query
            bypass_encoded: add woe encoding to the result
            template: 'td' for teradata or None
            nan_pattern_numbers: string value representing how to check nulls for numbers in SQL.
                For ex. "({0} IS NULL OR {0} = 'NaN')"
            nan_pattern_category: string value representing how to check nulls for categories in SQL.
            preprocessing: due to possible difference in schemes between SQL database and csv file user may
                specify dict how to preprocess each feature. For ex. if feature Feat_0 was treated as integer by
                model, but is actually string in database schema, you may pass
                preprocessing = {'Feat_0': CAST({0} as INTEGER)}

        Returns:
            sql query string.

        """
        return get_sql_inference_query(
            self,
            table_name,
            round_digits,
            round_features,
            output_name,
            alias,
            bypass_encoded,
            template,
            nan_pattern_numbers,
            nan_pattern_category,
            preprocessing,
            self._features_mark_values,
        )
