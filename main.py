import math
import numpy as np
import pandas as pd

from typing import Any
from pandas import DataFrame


class Node:
    def __init__(self, name: Any):
        self.name = name
        self.children: dict[Any, Node] = dict()

    def add_child(self, value, node) -> None:
        self.children[value] = node

    def __str__(self, level=0):
        indentation = "    " * level
        ret = f"{self.name}\n" if level == 0 else "\n"
        for value, child in self.children.items():
            ret += f"{indentation}└── '{str(value)}' ── {child.name}"
            ret += f"{child.__str__(level + 4)}"
        return ret


class DecisionTree:
    def __init__(self, ds: DataFrame):
        self._ds = ds
        self.tree = None

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
    def _get_subset(ds, attr, value):
        return ds[ds[attr] == value].reset_index(drop=True)

    def _build_tree(self, ds: DataFrame | None = None):
        if ds is None:
            ds = self._ds
        attr = self._find_optimal_attr(ds)
        tree = Node(attr)
        for value in np.unique(ds[attr]):
            subset = self._get_subset(ds, attr, value)
            output_values, counts = np.unique(self.__get_output_column(subset), return_counts=True)
            if len(counts) == 1:
                tree.add_child(value, Node(output_values[0]))
            else:
                subtree = self._build_tree(subset)
                tree.add_child(value, subtree)
        return tree

    def train(self):
        self.tree = self._build_tree()

    def predict(self, test_data):
        def recursive_predict(node, data):
            if not node.children:
                return node.name
            else:
                attr_value = data[node.name]
                if attr_value in node.children:
                    return recursive_predict(node.children[attr_value], data)
                else:
                    return "Unknown"

        if self.tree is None:
            raise ValueError("The decision tree has not been trained. Call the 'train' method first.")
        return recursive_predict(self.tree, test_data)


outlook = "overcast,overcast,overcast,overcast,rainy,rainy,rainy,rainy,rainy,sunny,sunny,sunny,sunny,sunny".split(",")
temp = "hot,cool,mild,hot,mild,cool,cool,mild,mild,hot,hot,mild,cool,mild".split(",")
humidity = "high,normal,high,normal,high,normal,normal,normal,high,high,high,high,normal,normal".split(",")
windy = "FALSE,TRUE,TRUE,FALSE,FALSE,FALSE,TRUE,FALSE,TRUE,FALSE,TRUE,FALSE,FALSE,TRUE".split(",")
play = "yes,yes,yes,yes,yes,yes,no,yes,no,no,no,no,yes,yes".split(",")
dataset = {"outlook": outlook, "temp": temp, "humidity": humidity, "windy": windy, "play": play}
df = pd.DataFrame(dataset, columns=["outlook", "temp", "humidity", "windy", "play"])

decision_tree = DecisionTree(ds=df)
decision_tree.train()
print(decision_tree.tree)

test_data = {"outlook": "sunny", "temp": "mild", "humidity": "normal", "windy": "TRUE"}
print(decision_tree.predict(test_data))

