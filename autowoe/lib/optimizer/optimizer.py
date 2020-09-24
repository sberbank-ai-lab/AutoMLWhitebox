import re

import lightgbm as lgb
import pandas as pd
import numpy as np

from typing import Any, Dict, Iterable, List, Tuple, Union
from abc import ABC, abstractmethod
from collections import OrderedDict
from itertools import product
from copy import deepcopy

from statsmodels.stats.proportion import proportion_confint

from autowoe.lib.utilities.utilities import code

np.random.seed(232)


class TreeParamOptimizer:
    """
    Класс для оптимизации гиперпараметров решающего дерева
    """

    def __init__(self, data: pd.DataFrame, params_range: Dict[str, tuple]):
        """
        Parameters
        ----------
        data
            Данные с признаком для которого производится биннинг (I колонка) и target (II колонка)
        params_range
             OrderedDict with parameters and ranges for binning algorithms
             params_range = OrderedDict({"max_depth": (4, 7, 17, 2, 3),  "min_child_samples": (40000, 20000, 5000),})
        """
        self._lgb_train = lgb.Dataset(data=pd.DataFrame(data.iloc[:, 0]), label=data.iloc[:, 1])
        self._params_range = params_range
        self._params_stats = None

    @property
    def __params_gen(self) -> Iterable[Tuple]:
        """
        :return:
        """
        return product(*self._params_range.values())

    def _get_score(self, params: Dict[str, Any], n: int) -> List[float]:
        """

        Parameters
        ----------
        params
            параметры дерева (которые мы ищем в этом классе)
        n
            количество кросс-валидаций для оценки гиперпараметров

        Returns
        -------

        """
        default_tree_params = {
            "boosting_type": "gbdt",
            "learning_rate": 1,
            "objective": "binary",
            "bagging_freq": 1,
            "bagging_fraction": 1,
            "feature_fraction": 1,
            "bagging_seed": 323,
            "n_jobs": 1,
            "verbosity": -1}
        unite_params = {**params, **default_tree_params}

        score = []
        for seed in range(n):
            cv_results = lgb.cv(params=unite_params, train_set=self._lgb_train, num_boost_round=1,
                                nfold=5, metrics='auc',
                                stratified=True, shuffle=True, seed=seed)
            score.append(cv_results["auc-mean"])
        return score

    def __get_stats(self, stats) -> None:
        """
        Функция вычисления статистик по scores
        Присутствуют три измерения. 0 - число комбинаций параметров, 1 - число кросс-валидаций, 2 - число фолдов к cv

        Parameters
        ----------
        stats

        Returns
        -------

        """
        stats = np.array(stats)
        median_, std_ = np.median(stats, axis=(1, 2)), np.std(stats, axis=(1, 2))

        id_max = zip(*(median_, -std_))  # номер комбинации с наилучшим качеством
        id_max = max(enumerate(id_max), key=lambda x: x[1])[0]

        stat_score = zip(*(median_, std_))
        self._params_stats = OrderedDict((key, value) for (key, value) in zip(self.__params_gen, stat_score)), id_max

    def __call__(self, n: int) -> Dict[str, Union[int, str, None]]:
        """

        Parameters
        ----------
        n
            Число перебираемых гиперпараметров

        Returns
        -------
        Параметры с наилучшим качеством (roc_auc) классификации
        """
        score_ = []
        for val in self.__params_gen:
            params = {key[1]: val[key[0]] for key in enumerate(self._params_range.keys())}
            score_.append(self._get_score(params, n))
        self.__get_stats(score_)

        opt_params = list(self._params_stats[0].keys())[self._params_stats[1]]
        return dict(zip(self._params_range.keys(), opt_params))


class StatOptimization:
    """
    Статистики для разбиения. Бутстреп
    Алгоритм пропускает категории если нет соответствующих значений в выборке
    """

    def __init__(self, data: pd.DataFrame, split: List[float], special_split):
        """

        Parameters
        ----------
        data
            Данные для кодирования. I колонка - признак, II - таргет
        split
            Разбиение [-7, 1.1, 2, 3, 12]
        special_split
            Специальные значения для кодирования
        """
        self.data = data
        self.split = split
        self.special_split = special_split
        self.__columns = data.columns

    @property
    def split(self):
        return self.__split

    @split.setter
    def split(self, split):
        if np.inf in split:
            self.__split = split
        else:
            self.__split = np.hstack((-np.inf, split))

    @property
    def __cat_encoding(self):
        """
        Кодирование признака с помощью self.split и self.special_split

        Returns
        -------

        """
        data = deepcopy(self.data)
        data = data.astype(float)

        categorial = np.searchsorted(self.split, data[self.__columns[0]], side="left")
        d_categorial = {x[0] + 1: x[1] for x in enumerate(self.split)}
        d_categorial[0] = -np.inf

        ind_special = data[self.__columns[0]].map(lambda x: x in self.special_split)

        data_encode = deepcopy(data)
        data_encode[self.__columns[0]] = categorial  # кодирование split
        data_encode[self.__columns[0]] = data_encode[self.__columns[0]].map(d_categorial)
        data_encode[self.__columns[0]] = data_encode[self.__columns[0]].astype(str) + "__real"
        data_encode.loc[ind_special, self.__columns[0]] = ((data.loc[ind_special, self.__columns[0]]).astype(str)
                                                           + "__real_u")
        # кодирование special_split
        return data_encode  # Кодирование не уникальных вещественных бинов осуществляется по левой границе

    def __get_special_v(self, val):
        """
        Parameters
        ----------
        val

        Returns
        -------

        """
        l_ = np.searchsorted(self.split, val, side="left")
        return [self.split[l_ - 1]]  # TODO ref + check !!!!

    @property
    def _get_border_dict(self) -> Dict[str, Tuple[str, str]]:
        """
        Словарь смежностей категорий

        Returns
        -------

        """
        border_dict = OrderedDict()
        for i in range(len(self.split)):  # заполнять сразу
            key = code(self.split[i], "real")
            border_dict[key] = []

            if i != 0:
                border_dict[key].append(code(self.split[i - 1], "real"))
            if i != len(self.split) - 1:
                border_dict[key].append(code(self.split[i + 1], "real"))
        # Добавили смежности обычных категорий

        for i in range(len(self.special_split)):
            key = code(self.special_split[i], "real_u")
            splits_ = [code(x, "real") for x in self.__get_special_v(self.special_split[i])]
            border_dict[key] = splits_
            for key_ in splits_:
                border_dict[key_].append(key)
                # Добавили смежности специальных категорий

        return border_dict

    def __call__(self):
        """
        Подсчет статистик для бинов

        Returns
        -------

        bootstrap_data -- Данные бутстрепа,
        border_dict -- словарь смежностей (какие категории имеют общую границу)
        """
        df = self.__cat_encoding
        stat_data = pd.crosstab(df[self.__columns[0]], df[self.__columns[1]])
        return stat_data, self._get_border_dict


class Bins:
    """
    Merge бинов. Изи реализация (Без пересчета после каждого merge статистики бутстрепом)
    """

    def __init__(self, sopt: StatOptimization, max_bin_count: int):
        """

        Parameters
        ----------
        sopt
            Стат инфа (сколько нулей и 1 в каждом бине) и матрица смежностей алгоритма
        max_bin_count
             Максимальное число бинов
        """
        self.stat_data, self.border_dict = sopt()
        self.max_bin_count = max_bin_count
        self.bin_count = self.stat_data.shape[0]

    def is_merge_true_(self, b_id0: str, b_id1: str):
        """
        Объединение двух бинов

        Parameters
        ----------
        b_id0
            id первого бина для объединения
        b_id1
            id воторго бина для объединения

        Returns
        -------

        """
        self.bin_count -= 1
        # 1. # border_dict
        ##########################################################################
        # Пополнение границ, с которыми пересекается объединяемый бин
        l_ = self.border_dict[b_id0]
        l_.remove(b_id1)
        l_.extend(self.border_dict[b_id1])
        l_.remove(b_id0)
        self.border_dict[b_id0] = list(set(l_))

        # 2. border_dict
        ##########################################################################
        # Удалить ключ id1
        self.border_dict.pop(b_id1)

        for key in self.border_dict:  # Удалить b_id1 и зменить их b_id0 из всех значений (списков)
            if b_id1 in self.border_dict[key]:
                self.border_dict[key].remove(b_id1)
                self.border_dict[key].append(b_id0)


class AbstractMerge(ABC):
    @staticmethod
    def _opt_init(train_f, split: List[float], max_bin_count: int) -> Bins:
        params = {
            "data": train_f,
            "split": split,
            "special_split": [], }

        params_q = {"sopt": StatOptimization(**params),
                    "max_bin_count": max_bin_count, }
        return Bins(**params_q)

    @abstractmethod
    def __call__(self):
        """
        Returns
        -------

        """
        pass


class BinMerge(AbstractMerge):
    """
    Класс для объединения бинов на основе доверительных интервалов
    """

    def __init__(self, alpha: float, train_f, split: List[float], max_bin_count: int):
        """
        Parameters
        ----------
        alpha
        train_f
        split
        max_bin_count
        """
        self.__alpha = alpha
        self._q_opt = self._opt_init(train_f=train_f, split=split, max_bin_count=max_bin_count)
        self._nearest_bin_length = self.__init_nearest

    @property
    def __init_nearest(self):  # инициализатор начального "расстояния" между бинами
        """
        Инициализация струтуры данных для порядкого преребора

        Returns
        -------

        """
        if self._q_opt.stat_data.shape[1] == 2:
            self.__total_bad, self.__total_good = self._q_opt.stat_data[1].sum(), self._q_opt.stat_data[0].sum()
        elif 1 in self._q_opt.stat_data.columns:
            self.__total_bad, self.__total_good = self._q_opt.stat_data[1].sum(), 0.5
        else:
            self.__total_bad, self.__total_good = 0.5, self._q_opt.stat_data[0].sum()

        nearest_bin_length = OrderedDict()
        for key0 in self._q_opt.border_dict.keys():
            for key1 in self._q_opt.border_dict[key0]:
                key = tuple(sorted([key0, key1], key=lambda x: float(x.split("__")[0])))  # сортировка слева напрво
                if key not in nearest_bin_length:
                    nearest_bin_length[key] = self.__intersect_stat(key0, key1)

        nearest_bin_length = OrderedDict(sorted(nearest_bin_length.items(), key=lambda x: -x[1]))

        return nearest_bin_length  # плохо возможно лучше использовать другую структуру

    def __local_woe(self, mean_val: float, count: int) -> float:
        """
        Рассчет WoE значения в бине

        Parameters
        ----------
        mean_val
        count

        Returns
        -------

        """
        t_bad = max(0.5, int(mean_val * count))
        t_good = count - t_bad
        return np.log((t_bad / self.__total_bad) / (t_good / self.__total_good))

    def __get_stat_(self, b_id: str, alpha_val: float) -> Tuple[float, float]:
        """
        Возврат статистики на основе бутстрепа

        Parameters
        ----------
        b_id
        alpha_val

        Returns
        -------

        """
        stat = self._q_opt.stat_data.loc[b_id]
        count_ones, count = stat[1], stat.sum()
        wilson_interval = proportion_confint(count_ones, count, alpha=alpha_val, method='wilson')
        wilson_interval = (self.__local_woe(wilson_interval[0], count), self.__local_woe(wilson_interval[1], count))
        return wilson_interval

    def __intersect_stat(self, b_id0: str, b_id1: str) -> float:
        """

        Parameters
        ----------
        b_id0
        b_id1

        Returns
        -------

        """
        stat_id0 = self.__get_stat_(b_id0, self.__alpha)
        stat_id1 = self.__get_stat_(b_id1, self.__alpha)
        return self.__interval_intersect(stat_id0, stat_id1)

    @staticmethod
    def __interval_intersect(a: Tuple[float, float], b: Tuple[float, float]) -> float:
        """
        Вычисление пересечения отрезков a и b + расширение на отрицательный случай

        Parameters
        ----------
        a
        b

        Returns
        -------

        """
        a0, a1 = a
        b0, b1 = b
        return min(a1, b1) - max(a0, b0)

    def is_merge_true_(self, b_id0: str, b_id1: str):
        """
        Parameters
        ----------
        b_id0
        b_id1

        Returns
        -------

        """
        self._q_opt.is_merge_true_(b_id0, b_id1)
        # 1. __nearest_bin
        ##########################################################################
        # удалть id1 из ключей
        for key in list(self._nearest_bin_length.keys()):
            if b_id1 in key:
                del self._nearest_bin_length[key]

        # 2. __nearest_bin
        ##########################################################################
        # Добавление новых расстояний + пересчет расстояний в __nearest bin с b_id0
        self._q_opt.stat_data.loc[b_id0, :] = self._q_opt.stat_data.loc[b_id0] + self._q_opt.stat_data.loc[b_id1]
        for key in self._q_opt.border_dict[b_id0]:  # получить все пары из border_dict и итерироваться по ним
            key_ = tuple(sorted([b_id0, key], key=lambda x: float(x.split("__")[0])))  # сортировка слева напрво
            self._nearest_bin_length[key_] = self.__intersect_stat(*key_)
        # 3.__nearest_bin
        ##########################################################################
        # Отсоритровать расстояния
        self._nearest_bin_length = OrderedDict(sorted(self._nearest_bin_length.items(), key=lambda x: -x[1]))

    def merge_cycle(self):  # TODO Write tests !!!
        """
        Вспомогательная функция категорий
        Обход self.border_dict один раз и проверка на merge этих категорий

        Returns
        -------

        """
        if len(self._nearest_bin_length.values()) == 0:  # получился один бин
            return True

        max_bin_count = np.inf if self._q_opt.max_bin_count is None else self._q_opt.max_bin_count

        #  выход пока "расстояние" между бинами будет строго положительным и число бинов будет меньше max_bin_count
        while (list(self._nearest_bin_length.values())[-1] < 0) or (self._q_opt.bin_count > max_bin_count):
            if self._q_opt.bin_count <= 2:  # TODO from queue import PriorityQueue
                return True

            # print(self._nearest_bin_length.values())

            b_id0, b_id1 = list(self._nearest_bin_length.keys())[-1]
            self.is_merge_true_(b_id0, b_id1)

        return False

    def __call__(self):
        """
        Возврат разбиения вида [-np.inf, -7, 1.1, 2, 3, 12] + уникальные значения

        Returns
        -------

        """
        is_drop = self.merge_cycle()
        unique_split = deepcopy(self._q_opt.border_dict)

        split = list(filter(re.compile(".*real$").match, unique_split))
        split = np.array([x[:-6] for x in split]).astype(float)

        return split, is_drop


class FMerge(AbstractMerge):
    """
    Класс для объединения бинов на основе доверительных интервалов
    """

    def __init__(self, cv_index_split: Dict[int, List[int]], train_f: pd.DataFrame, split: List[float],
                 max_bin_count: int):
        """
        Parameters
        ----------
        cv_index_split
        train_f
        split
        max_bin_count
        """
        self._q_opt = self._opt_init(train_f=train_f, split=split, max_bin_count=max_bin_count, )
        self._folds_opt = self.__folds_opt_init(train_f=train_f, cv=cv_index_split, split=split,
                                                max_bin_count=max_bin_count)
        self.__total_bad = train_f.iloc[:, 1].sum()
        self.__total_good = train_f.shape[0] - self.__total_bad
        self.__nearest_bin_length = self.__init_nearest

    @property
    def __init_nearest(self) -> Dict[Tuple[str, str], float]:
        """
        Инициализатор начального "расстояния" между бинами
        Инициализация струтуры данных для порядкого преребора

        Returns
        -------

        """
        if self._q_opt.stat_data.shape[1] == 2:
            self.__total_bad, self.__total_good = self._q_opt.stat_data[1].sum(), self._q_opt.stat_data[0].sum()
        elif 1 in self._q_opt.stat_data.columns:
            self.__total_bad, self.__total_good = self._q_opt.stat_data[1].sum(), 0.5
        else:
            self.__total_bad, self.__total_good = 0.5, self._q_opt.stat_data[0].sum()

        nearest_bin = OrderedDict()
        for key0 in self._q_opt.border_dict.keys():
            for key1 in self._q_opt.border_dict[key0]:
                key = tuple(sorted([key0, key1], key=lambda x: float(x.split("__")[0])))  # сортировка слева напрво
                if key not in nearest_bin:
                    nearest_bin[key] = self.__intersect_stat(key0, key1)

        nearest_bin_length = OrderedDict(
            sorted(nearest_bin.items(), key=lambda x: -x[1]))  # TODO ref from queue import PriorityQueue
        return nearest_bin_length  # плохо возможно лучше использовать другую структуру

    def __folds_opt_init(self, train_f: pd.DataFrame, cv: Dict[int, List[int]], split: List[float],
                         max_bin_count: int) -> Dict[int, Bins]:
        """

        Parameters
        ----------
        train_f
        cv
        split
        max_bin_count

        Returns
        -------

        """
        folds_opt = dict()
        for key in cv:
            folds_opt[key] = self._opt_init(train_f.iloc[cv[key], :], split, max_bin_count)
        return folds_opt

    def _local_woe(self, t_good: int, t_bad: int, folds: bool = False) -> float:
        """

        Parameters
        ----------
        t_good
        t_bad
        folds

        Returns
        -------

        """
        if folds:
            c = (len(self._folds_opt) - 1) / len(self._folds_opt)
        else:
            c = 1
        t_good = max(0.5, t_good)  # почти честное WoE (для фолдов)
        return np.log((t_bad / (c * self.__total_bad)) / (t_good / (c * self.__total_good)))

    @staticmethod
    def __stat(stat_main_id0: float, stat_main_id1: float, stat_folds_id1: np.array) -> float:
        """


        Parameters
        ----------
        stat_main_id0
        stat_main_id1
        stat_folds_id1

        Returns
        -------

        """
        stat_diff = (stat_main_id1 - stat_main_id0) * (stat_folds_id1 - stat_main_id0)
        if all(stat_diff > 0):
            return 0
        else:
            return -np.sum(np.minimum(stat_diff, 0))

    def __intersect_stat(self, b_id0: str, b_id1: str) -> float:
        """


        Parameters
        ----------
        b_id0: str
        b_id1: str

        Returns
        -------

        """
        stat_main_id0 = self._local_woe(*self._q_opt.stat_data.loc[b_id0].values)
        stat_main_id1 = self._local_woe(*self._q_opt.stat_data.loc[b_id1].values)
        stat_folds_id0 = np.array(
            [self._local_woe(*self._folds_opt[key].stat_data.loc[b_id0].values, folds=True) for key in self._folds_opt])
        stat_folds_id1 = np.array(
            [self._local_woe(*self._folds_opt[key].stat_data.loc[b_id1].values, folds=True) for key in self._folds_opt])
        diff = min(self.__stat(stat_main_id0, stat_main_id1, stat_folds_id1), self.__stat(stat_main_id1, stat_main_id0,
                                                                                          stat_folds_id0))
        return diff

    def is_merge_true_(self, b_id0: str, b_id1: str) -> None:
        """

        Parameters
        ----------
        b_id0
        b_id1

        Returns
        -------

        """
        self._q_opt.is_merge_true_(b_id0, b_id1)
        [self._folds_opt[key].is_merge_true_(b_id0, b_id1) for key in self._folds_opt]
        # 3. __nearest_bin
        ##########################################################################
        # удалть id1 из ключей
        for key in list(self.__nearest_bin_length.keys()):
            if b_id1 in key:
                del self.__nearest_bin_length[key]

        # 4. __nearest_bin
        ##########################################################################
        # Добавление новых расстояний + пересчет расстояний в __nearest bin с b_id0
        self._q_opt.stat_data.loc[b_id0, :] = (self._q_opt.stat_data.loc[b_id0] +
                                               self._q_opt.stat_data.loc[b_id1])
        for key in self._folds_opt:
            self._folds_opt[key].stat_data.loc[b_id0, :] = (self._folds_opt[key].stat_data.loc[b_id0] +
                                                            self._folds_opt[key].stat_data.loc[b_id1])
        for key in self._q_opt.border_dict[b_id0]:  # получить все пары из border_dict и итерироваться по ним
            key_ = tuple(sorted([b_id0, key], key=lambda x: float(x.split("__")[0])))  # сортировка слева напрво
            self.__nearest_bin_length[key_] = self.__intersect_stat(*key_)

        # 5.__nearest_bin
        ##########################################################################
        # Отсоритровать расстояния
        self.__nearest_bin_length = OrderedDict(sorted(self.__nearest_bin_length.items(), key=lambda x: -x[1]))

    def merge_cycle(self):  # Отрефакторить и исправить !!!
        """
        Вспомогательная функция категорий
        Обход self.border_dict один раз и проверка на merge этих категорий

        Returns
        -------

        """
        if len(self.__nearest_bin_length.values()) == 0:  # получился один бин
            return True

        max_bin_count = np.inf if self._q_opt.max_bin_count is None else self._q_opt.max_bin_count

        #  выход пока "расстояние" между бинами будет строго положительным и число бинов будет меньше max_bin_count
        while (list(self.__nearest_bin_length.values())[-1] < 0) or (self._q_opt.bin_count > max_bin_count):
            if self._q_opt.bin_count <= 2:  # TODO from queue import PriorityQueue
                return True
            b_id0, b_id1 = list(self.__nearest_bin_length.keys())[-1]
            self.is_merge_true_(b_id0, b_id1)
        return False

    def __call__(self) -> Tuple[np.ndarray, bool]:
        """
        Возврат разбиения вида [-7, 1.1, 2, 3, 12] +
        + уникальные значения
        Returns
        -------

        """
        is_drop = self.merge_cycle()
        unique_split = deepcopy(self._q_opt.border_dict)

        split = list(filter(re.compile(".*real$").match, unique_split))
        split = np.array([x[:-6] for x in split]).astype(float)

        return split, is_drop
