import collections

import pandas as pd

from typing import Dict, Hashable, Optional, Any
from copy import deepcopy

from .features_checkers_handlers import dates_handler, dates_checker, cat_checker


class TypesHandler:
    """
    Класс для автоматического определения типов признаков.
    Базовая имплементация порядка разработки:

    0.
        0.a) Парсим то, что указал юзер
        0.b) Даты пасим С указанием сезонности ("m", "d", "wd", "h", "min")
        (месяц, день, день недели, час, минута)
    1.
        Если стринга, то категория
    2.
        Если отношение shape[1] к количеству уникальных значений >> 5, то категория
    """

    def __init__(self,
                 train: pd.DataFrame,
                 public_features_type: Dict[Hashable, Any],
                 max_bin_count: Dict[Hashable, Optional[int]] = None,
                 features_monotone_constraints: Optional[dict] = None):
        """

        Args:
            train:
            public_features_type:
            max_bin_count:
            features_monotone_constraints:
        """
        self.__train = deepcopy(train)
        self.__public_features_type = deepcopy(public_features_type)
        self.__private_features_type = dict()

        if max_bin_count is None:
            max_bin_count = {}
        self.__max_bin_count = collections.defaultdict(lambda: None, max_bin_count)

        if features_monotone_constraints is None:
            features_monotone_constraints = {}
        self.__features_monotone_constraints = collections.defaultdict(lambda: "0", features_monotone_constraints)

    @property
    def train(self):
        """
        Read only

        Return:

        """
        return self.__train

    @property
    def public_features_type(self):
        """
        Read only

        Return:

        """
        return self.__public_features_type

    @property
    def private_features_type(self):
        """
        Read only

        Returns:

        """
        return self.__private_features_type

    @property
    def max_bin_count(self):
        """

        Returns:

        """
        return self.__max_bin_count

    @property
    def features_monotone_constraints(self):
        """

        Returns:

        """
        return self.__features_monotone_constraints

    def __feature_handler(self, feature_name):
        """

        Args:
            feature_name:

        Returns:

        """
        if dates_checker(self.__train[feature_name]):
            new_features, feature_type = dates_handler(self.__train[feature_name])
            self.__public_features_type[feature_name] = feature_type
            for new_feature_name, new_feature in new_features:
                self.__train[new_feature_name] = new_feature
                self.__max_bin_count[new_feature_name] = self.max_bin_count[feature_name]
                self.__private_features_type[new_feature_name] = "real"
                self.__features_monotone_constraints[new_feature_name] = \
                    self.features_monotone_constraints[feature_name]

        elif cat_checker(self.__train[feature_name]):
            self.__public_features_type[feature_name] = "cat"
            self.__private_features_type[feature_name] = "cat"
            self.__features_monotone_constraints[feature_name] = "1"
        else:
            self.__public_features_type[feature_name] = "real"
            self.__private_features_type[feature_name] = "real"

    def transform(self):
        """
        Основной метод данного класса.
        Если feature_type[feature] == None, то парсим тип признкака
        Иначе происходит обработка указанных типов.
        Возмоожные типы признаков:
            "cat"
            "real"
            ("%Y%d%m", ("m", "d", "wd", "h", "min"))

        Returns:

        """
        for feature_name in self.public_features_type:
            if not self.public_features_type[feature_name]:
                self.__feature_handler(feature_name)
            elif isinstance(self.public_features_type[feature_name], tuple):  # переданы данные для дат
                new_features, _ = dates_handler(self.train[feature_name], self.public_features_type[feature_name])
                for new_feature_name, new_feature in new_features:
                    self.__train[new_feature_name] = new_feature
                    self.__max_bin_count[new_feature_name] = self.max_bin_count[feature_name]
                    self.__private_features_type[new_feature_name] = "real"
                    self.__features_monotone_constraints[new_feature_name] = \
                        self.__features_monotone_constraints[feature_name]

            elif self.public_features_type[feature_name] == "cat":
                self.__private_features_type[feature_name] = "cat"
                self.__features_monotone_constraints[feature_name] = "1"

            elif self.public_features_type[feature_name] == "real":
                self.__private_features_type[feature_name] = "real"
                self.__train[feature_name] = pd.to_numeric(self.train[feature_name], errors="coerce")

            else:
                raise ValueError("The specified data type is not supported")

        return (self.train, self.public_features_type, self.private_features_type,
                self.max_bin_count, self.features_monotone_constraints)
