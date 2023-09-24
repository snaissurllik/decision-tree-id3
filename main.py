import math
import numpy as np
import pandas as pd
import pprint

from pandas import DataFrame


class Node:
    def __init__(self, name: str):
        self.name = name
        self.left = None
        self.right = None


class DecisionTree:
    def __init__(self, ds: DataFrame):
        self._ds = ds

    @staticmethod
    def __get_output_column(ds: DataFrame):
        return ds[ds.keys()[-1]]

    @staticmethod
    def _get_ds_attrs(ds: DataFrame):
        return ds.keys()[:-1]

    def _calc_entropy(self, ds) -> float:
        entropy = 0
        output_column = self.__get_output_column(ds)
        for value in output_column.unique():
            p = output_column.value_counts()[value] / len(output_column)
            entropy -= p * math.log2(p) if p > 0 else 0
        return entropy

    def __calc_info_gain_by_attr(self, attr: str, ds: DataFrame) -> float:
        info_gain = self._calc_entropy(ds)
        attr_values = ds[attr].unique()
        for attr_value in attr_values:
            subset = ds[ds[attr] == attr_value]
            info_gain -= ds[ds[attr] == attr_value].shape[0] / ds.shape[0] * self._calc_entropy(subset)
        return info_gain

    def _find_optimal_attr(self, ds: DataFrame):
        attrs = self._get_ds_attrs(ds)
        info_gains = [self.__calc_info_gain_by_attr(attr, ds) for attr in attrs]
        return attrs[np.argmax(info_gains)]

    @staticmethod
    def _get_subset(ds, node, value):
        return ds[ds[node] == value].reset_index(drop=True)

    def _build_tree(self, ds: DataFrame | None = None, tree=None):
        if ds is None:
            ds = self._ds
        node = self._find_optimal_attr(ds)
        attr_values = np.unique(ds[node])
        if tree is None:
            tree = {node: {}}
        for value in attr_values:
            subset = self._get_subset(ds, node, value)
            unique_values, counts = np.unique(self.__get_output_column(subset), return_counts=True)
            if len(counts) == 1:  # Checking purity of subset
                tree[node][value] = unique_values[0]
            else:
                tree[node][value] = self._build_tree(subset)  # Calling the function recursively
        return tree

    def train(self):
        return self._build_tree()


outlook = "overcast,overcast,overcast,overcast,rainy,rainy,rainy,rainy,rainy,sunny,sunny,sunny,sunny,sunny".split(",")
temp = "hot,cool,mild,hot,mild,cool,cool,mild,mild,hot,hot,mild,cool,mild".split(",")
humidity = "high,normal,high,normal,high,normal,normal,normal,high,high,high,high,normal,normal".split(",")
windy = "FALSE,TRUE,TRUE,FALSE,FALSE,FALSE,TRUE,FALSE,TRUE,FALSE,TRUE,FALSE,FALSE,TRUE".split(",")
play = "yes,yes,yes,yes,yes,yes,no,yes,no,no,no,no,yes,yes".split(",")

dataset = {"outlook": outlook, "temp": temp, "humidity": humidity, "windy": windy, "play": play}
df = pd.DataFrame(dataset, columns=["outlook", "temp", "humidity", "windy", "play"])
decision_tree = DecisionTree(ds=pd.read_csv("diabetes_dataset.csv"))
t = decision_tree.train()
pprint.pprint(t)
