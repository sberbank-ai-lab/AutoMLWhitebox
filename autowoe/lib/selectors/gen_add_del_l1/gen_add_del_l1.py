import lightgbm as lgb
import pandas as pd
import numpy as np

from copy import deepcopy
from typing import Union, Dict, List, Tuple, TypeVar, Optional, Generator

from joblib import Parallel, delayed
from sklearn.metrics import roc_auc_score

from ..utilities.utilities_sklearn import Result, l1_select
from ...utilities.eli5_permutation import get_score_importances

WoE = TypeVar("WoE")
feature = Union[str, int, float]
f_list_type = List[feature]


class GenAddDelL1:
    """
    Класс для отбора признаков. Реализация алгоритма GenAddDelL1
    Данный подход является смесью Генетического алгоритма, алгоритма AddDel и отбора Lasso.
    """

    def __init__(self, interpreted_model: bool, train: pd.DataFrame, target: pd.Series, n_jobs: int,
                 cv_split: Dict[int, Tuple[List[int], List[int]]], group_kf: Optional[np.ndarray],
                 population_size: int, feature_groups_count, imp_type: str):
        """

        Parameters
        ----------
        interpreted_model
           флаг интерпретирумости модели. Под интерпетируемостью в смысле библиотеки понимается достижения
           отрицательных регрессионных коэффициентов

        train
            предобработанная выборка для обучения
        target
           колонка с целевой переменной
        n_jobs
            Число используемых ядер
        cv_split

        group_kf

        population_size: int
            Количество комбинаций признаков в очередной группе. Общее число групп = feature_groups_count


        feature_groups_count: int

        imp_type: str
           Тип важности признаков. От него зависит порядок признаков, а следовательно то, в какую группу попадет.
           Также в части алгоритма add_del эти значения используются в вероятности добавления / удаления из набора
        """
        self.train = train
        self.target = target

        self.__interpreted_model = interpreted_model
        self.__n_jobs = n_jobs
        self.__features = train.columns
        self.__cv_split = cv_split
        self.__group_kf = group_kf
        self.__population_size = population_size
        self.__feature_groups_count = feature_groups_count
        self.__imp_type = imp_type

    @staticmethod
    def get_feature_sample(feature_weights: pd.Series) -> f_list_type:
        """

        Parameters
        ----------
        feature_weights: pd.Series

        Returns
        -------

        """
        while True:
            x_ = deepcopy(feature_weights)
            x_ = x_.map(lambda x: np.random.choice(a=2, size=1, p=[1 - x, x])[0])
            feature_sample = list(x_.loc[x_ == 1].index)
            if len(feature_sample) > 0:
                return feature_sample

    @staticmethod
    def __cross(features_1: f_list_type, features_2: f_list_type) -> f_list_type:
        """
        Скрещивание двух наборов признаков

        Parameters
        ----------
        features_1:
            Первый набор признаков для скрещивания
        features_2:
            Второй набор признаков для скрещивания
        Returns
        -------
        Новый набор признаков

        """
        while True:
            delta = 0.1 * np.random.random_sample()
            choose_1 = np.random.choice(a=2, size=len(features_1), replace=True, p=[0.1 - delta, 0.9 + delta])
            choose_2 = np.random.choice(a=2, size=len(features_2), replace=True, p=[0, 1])
            choose_1, choose_2 = dict(zip(features_1, choose_1)), dict(zip(features_2, choose_2))
            choose_1 = [key for key in choose_1 if choose_1[key] == 1]
            choose_2 = [key for key in choose_2 if choose_2[key] == 1]
            choose = [*choose_1, *choose_2]
            if len(choose) != 0:
                return list(set(choose))

    def __genetic_cross(self, best_features: List[f_list_type], feature_weights, n_repeat: int) -> List[f_list_type]:
        """
        Стадия мутации на основе скрещивания

        Parameters
        ----------
        best_features

        feature_weights

        n_repeat

        Returns
        -------

        """
        new_features = deepcopy(best_features)
        for features in best_features:
            for _ in range(n_repeat):
                new_features.append(self.__cross(features, self.get_feature_sample(feature_weights)))
        return new_features

    @staticmethod
    def __split(feature_weights: pd.Series, feature_groups_count: int) -> Generator[pd.Series, None, None]:
        """
        Parameters
        ----------
        feature_weights

        feature_groups_count

        Returns
        -------

        """
        _feature_weights = feature_weights.sort_values(ascending=False)
        if _feature_weights.shape[0] <= feature_groups_count:
            raise ValueError("The features quantity < feature_groups_count")

        index_split = np.array_split(_feature_weights, feature_groups_count)
        return (x for x in index_split)

    def __genetic_estimation(self, dataset: Tuple[pd.DataFrame, pd.DataFrame],
                             cv_split: Dict[int, Tuple[np.array, np.array]], 
                             l1_base_step: int, l1_exp_step: float, 
                             early_stopping_rounds: int, features: List[f_list_type],
                             auc_tol: float = 1e-4) -> Tuple[List[float], List[Tuple[f_list_type, Result]]]:
        """
        Стадия оценивания

        Parameters
        ----------
        dataset

        cv_split

        l1_base_step

        early_stopping_rounds

        features

        Returns
        -------

        """
        # Фикс от Антона) 
        _n_jobs_gen = min(self.__n_jobs, self.__population_size)
        
        if self.__population_size * len(cv_split) <= self.__n_jobs:
            _n_jobs_l1 = len(cv_split)
        else:
            _n_jobs_l1 = max(self.__n_jobs // self.__population_size, 1)

        
        datasets = ((self.__interpreted_model,
                     _n_jobs_l1,
                     (deepcopy(dataset[0].loc[:, sample]), deepcopy(self.target)),
                     l1_base_step,
                     l1_exp_step,
                     early_stopping_rounds,
                     cv_split,
                     False, 
                     auc_tol) for sample in features)

        if self.__n_jobs > 1:
            with Parallel(n_jobs=_n_jobs_gen, prefer="processes") as p:
                result = p(delayed(l1_select)(*i) for i in datasets)
        else:
            result = []
            for i in datasets:
                result.append(l1_select(*i))
        scores = [x[1].score for x in result]
        return scores, result

    @staticmethod
    def __genetic_selection(population_size: int, scores: List[float],
                            result: List[Tuple[List[f_list_type], Result]]) -> Tuple[List[f_list_type], List[float]]:
        """
        Parameters
        ----------
        population_size

        scores

        result

        Returns
        -------

        """
        # TODO:  check len(int) != 0
        size = int(population_size ** 0.5)
        ind = sorted(range(len(scores)), key=lambda i: scores[i])[-size:]
        best_features = [result[i][0] for i in ind]
        best_scores = [result[i][1][0] for i in ind]

        not_empty_features = [i for i, x in enumerate(best_features) if len(x) != 0]
        best_features = [best_features[i] for i in not_empty_features]
        best_scores = [best_scores[i] for i in not_empty_features]
        return best_features, best_scores

    @staticmethod
    def __genetic_stopping(best_scores: List[float], best_features: List[f_list_type],
                           best_features_: Optional[List[f_list_type]], max_score: float,
                           counter: int) -> Tuple[int, List[f_list_type], float]:
        """

        Parameters
        ----------
        best_scores

        best_features

        best_features_

        max_score

        counter

        Returns
        -------

        """
        mean_score = np.mean(best_scores)
        print(f"Mean cv score in the step = {mean_score}")
        if mean_score > max_score:
            max_score = mean_score
            best_features_ = deepcopy(best_features)
            counter = 0
        else:
            counter += 1
        return counter, best_features_, max_score

    def __genetic_run(self, feature_weights: pd.Series, population_size: int,
                      feature_groups_count: int, dataset: Tuple[pd.DataFrame, pd.DataFrame],
                      new_cv_index_split: Dict[int, Tuple[np.array, np.array]], 
                      l1_base_step: int, l1_exp_step: float,
                      early_stopping_rounds: int, auc_tol: float = 1e-4) -> List[f_list_type]:
        """
        Parameters
        ----------
        feature_weights

        population_size

        feature_groups_count

        dataset

        new_cv_index_split

        l1_base_step

        early_stopping_rounds


        Returns
        -------

        """
        _feature_weights = feature_weights.rank() / feature_weights.shape[0]
        _feature_weights = np.maximum(_feature_weights, 0.05)
        _feature_weights = np.minimum(_feature_weights, 0.95)
        counter = 0
        max_score = 0
        best_features_ = None  # Лучшие наборы признаков
        ##################################################################################################
        _feature_weights_split = self.__split(_feature_weights, feature_groups_count)
        ##################################################################################################
        # генерация
        f_split = next(_feature_weights_split)
        features = (self.get_feature_sample(f_split) for _ in range(population_size))
        ##################################################################################################
        while True:
            # оценивание
            scores, result = self.__genetic_estimation(dataset, new_cv_index_split, l1_base_step, l1_exp_step,
                                                       early_stopping_rounds,
                                                       features, auc_tol)
            ##################################################################################################
            # селекция
            best_features, best_scores = self.__genetic_selection(population_size, scores, result)
            ##################################################################################################
            # критерий остановы
            counter, best_features_, max_score = self.__genetic_stopping(best_scores, best_features, best_features_,
                                                                         max_score, counter)
            if counter > early_stopping_rounds:
                return best_features_
            ##################################################################################################
            # мутация
            try:
                f_split = next(_feature_weights_split)
                features = self.__genetic_cross(best_features_, f_split, int(population_size ** 0.5))  # TODO: ref !!
            except StopIteration:
                return best_features_

    def __call__(self, features_fit: f_list_type,
                 l1_base_step: float,
                 l1_exp_step: float,
                 early_stopping_rounds: int,
                 auc_tol: float = 1e-4,
                 feature_history: Dict[str, str] = None
                ) -> Tuple[f_list_type, Result]:
        """
        Точка входа в постотбор признаков

        Parameters
        ----------
        features_fit: f_list_type

        l1_base_step: float
            Базовый шаг уведичения коэффициента в l1 регулризации

        l1_exp_step: float
            Нелинейный коэффициент увеличения шага в l1 регуляризации

        early_stopping_rounds:

        Returns
        -------

        """
        features_fit_ = features_fit.copy()
        dataset = self.train[features_fit_], self.target

        train_ind, test_ind = self.__cv_split[0][0], self.__cv_split[0][1]
        x_train, x_test, y_train, y_test = (dataset[0].iloc[train_ind, :], dataset[0].iloc[test_ind, :],
                                            dataset[1].iloc[train_ind], dataset[1].iloc[test_ind])

        params = {
            "boosting_type": "gbdt",
            "objective": "binary",
            'nthread': self.__n_jobs,
            "bagging_seed": 323,
            "metric": "auc",
            "min_gain_to_split": 0.01,
            "verbosity": -1}

        if self.__imp_type == "feature_imp":
            lgb_train = lgb.Dataset(data=x_train, label=y_train)
            lgb_test = lgb.Dataset(data=x_test, label=y_test)
            clf = lgb.train(params=params, train_set=lgb_train, verbose_eval=False,
                            early_stopping_rounds=10, valid_sets=lgb_test)
            feature_weights = pd.Series(clf.feature_importance(), x_train.columns)
        elif self.__imp_type == "perm_imp":
            clf = lgb.LGBMClassifier(**params)

            def score(x, y):
                y_pred = clf.predict_proba(x)[:, 1]
                return roc_auc_score(y, y_pred)

            clf.fit(X=x_train, y=y_train)
            _, score_decreases = get_score_importances(score, x_test.values, y_test)
            feature_weights = pd.Series(np.array(score_decreases).mean(axis=0), x_train.columns)
        else:
            raise ValueError("select_type is feature_imp or perm_imp")

        new_cv_split = deepcopy(self.__cv_split)
        out_of_fold_split = new_cv_split.pop(0)

        for key in new_cv_split:
            new_cv_split[key] = (np.setdiff1d(new_cv_split[key][0], out_of_fold_split[1]),
                                 new_cv_split[key][1])

        best_features = self.__genetic_run(feature_weights=feature_weights,
                                           population_size=self.__population_size,
                                           feature_groups_count=self.__feature_groups_count,
                                           dataset=dataset,
                                           new_cv_index_split=new_cv_split,
                                           l1_base_step=l1_base_step,
                                           l1_exp_step=l1_exp_step, 
                                           early_stopping_rounds=early_stopping_rounds, 
                                           auc_tol=auc_tol)
        # out of fold checking

        out_of_fold_result = []
        for features in best_features:
            out_of_fold_result.append(l1_select(self.__interpreted_model,
                                                self.__n_jobs,
                                                dataset=(dataset[0][features], dataset[1]),
                                                l1_base_step=l1_base_step,
                                                l1_exp_step=l1_exp_step,
                                                early_stopping_rounds=early_stopping_rounds,
                                                cv_split={0: out_of_fold_split},
                                                verbose=True, 
                                                auc_tol=auc_tol)[1])

        max_score_arg = np.argmax([x.score for x in out_of_fold_result])
        best_features, result = best_features[max_score_arg], out_of_fold_result[max_score_arg]

        final_dataset = self.train[best_features], self.target
        _, result = l1_select(self.__interpreted_model,  # TODO ref ?!
                              self.__n_jobs,
                              dataset=final_dataset,
                              l1_base_step=l1_base_step,
                              l1_exp_step=l1_exp_step,
                              early_stopping_rounds=early_stopping_rounds,
                              cv_split=self.__cv_split,
                              verbose=True, 
                              auc_tol=auc_tol)

        return best_features, result

