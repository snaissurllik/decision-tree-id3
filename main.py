import math
import numpy as np
import pandas as pd

from typing import Any
from pandas import DataFrame, Index, Series


class Node:
    """
    Class represents tree node of a decision tree.
    """

    def __init__(self, name: Any):
        self.name = name
        self.children: dict[Any, Node] = dict()

    def add_child(self, value, node) -> None:
        """
        Add a child node with a specified value to the relation.

        Args:
            value (Any): The value associated with the child node.
            node (Node): The child node to be added.

        Returns:
            None
        """

        self.children[value] = node

    def __str__(self, level: int = 0):
        """
        String representation of the tree.
        """

        indentation = "    " * level
        ret = f"{self.name}\n" if level == 0 else "\n"
        for value, child in self.children.items():
            ret += f"{indentation}└── '{str(value)}' ── {child.name}"
            ret += f"{child.__str__(level + 4)}"
        return ret


class ID3DecisionTree:
    """
    Implementation of the ID3 Decision Tree algorithm.
    """

    def __init__(self, ds: DataFrame, threshold: float = 0.8):
        self.__ds = ds
        self.__tree = None
        self.threshold = threshold
        self.output_column_name = self.__get_output_column_name(ds)

    @staticmethod
    def __get_output_column_name(ds: DataFrame) -> str:
        """
        Get the name of the output column in the dataset.

        Args:
            ds (DataFrame): The input dataset containing features and output labels.

        Returns:
            str: The name of the output column.
        """

        return ds.keys()[-1]

    def __get_output_column(self, ds: DataFrame) -> Series:
        """
        Get the output column from the dataset.

        Args:
            ds (DataFrame): The input dataset containing features and output labels.

        Returns:
            Series: The output column.
        """

        return ds[self.output_column_name]

    @staticmethod
    def __get_ds_features(ds: DataFrame) -> Index:
        """
        Get the names of the feature columns in the dataset.

        Args:
            ds (DataFrame): The input dataset containing features and output labels.

        Returns:
            Index: The feature column names.
        """

        return ds.keys()[:-1]

    @staticmethod
    def __get_ds_columns(ds: DataFrame) -> Index:
        """
        Get the names of all columns in the dataset.

        Args:
            ds (DataFrame): The input dataset containing features and output labels.

        Returns:
            Index: The column names.
        """

        return ds.keys()

    def __calc_entropy(self, ds) -> float:
        """
        Calculate the entropy of the output column in the dataset.

        Args:
            ds (DataFrame): The input dataset containing features and output labels.

        Returns:
            float: The calculated entropy value.
        """

        entropy = 0
        output_column = self.__get_output_column(ds)
        for value in output_column.unique():
            p = output_column.value_counts()[value] / len(output_column)
            entropy -= p * math.log2(p) if p > 0 else 0
        return entropy

    def __calc_info_gain_by_feature(self, feature: str, ds: DataFrame) -> float:
        """
        Calculate the information gain for a given feature in the dataset.

        Args:
            feature (str): The name of the feature column.
            ds (DataFrame): The input dataset containing features and output labels.

        Returns:
            float: The calculated information gain value.
        """

        info_gain = self.__calc_entropy(ds)
        feature_values = ds[feature].unique()
        for feature_value in feature_values:
            subset = ds[ds[feature] == feature_value]
            info_gain -= ds[ds[feature] == feature_value].shape[0] / ds.shape[0] * self.__calc_entropy(subset)
        return info_gain

    def __find_optimal_feature(self, ds: DataFrame) -> str | None:
        """
        Find the optimal feature for splitting the dataset.

        Args:
            ds (DataFrame): The input dataset containing features and output labels.

        Returns:
            str | None: The name of the optimal feature.
        """

        if (features := self.__get_ds_features(ds)).empty:
            return
        info_gains = [self.__calc_info_gain_by_feature(feature, ds) for feature in features]
        return features[np.argmax(info_gains)]

    @staticmethod
    def __get_subset(ds, feature, value):
        """
        Get a subset of the dataset where the specified feature has a given value.

        Args:
            ds (DataFrame): The input dataset containing features and output labels.
            feature (str): The name of the feature column.
            value: The value to filter the feature column.

        Returns:
            The subset of the dataset.
        """

        ds = ds[ds[feature] == value].reset_index(drop=True)
        return ds.drop(feature, axis=1)

    def __get_output_value_threshold(self, values, counts):
        """
        Get the output value based on a threshold condition.

        Args:
            values: Iterable of unique output values.
            counts: Iterable of counts corresponding to each unique output value.

        Returns:
            The selected output value or None.
        """

        total_count = sum(counts)
        max_count_value, max_count = values[0], counts[0]
        for value, count in zip(values, counts):
            if count > max_count:
                max_count_value = value
                max_count = count
        return max_count_value if max_count / total_count >= self.threshold else None

    def __get_most_common_output_value(self, ds: DataFrame):
        """
        Get the most common output value in the given dataset.

        Args:
            ds (DataFrame): The input dataset containing output labels.

        Returns:
            Any: The most common output value in the dataset.
        """

        return self.__get_output_column(ds).mode()[0]

    def __build_tree(self, ds: DataFrame | None = None):
        """
        Recursively build the decision tree from the dataset.

        Args:
            ds (DataFrame | None, optional): The dataset to use for building the tree.
            If None, the object's dataset will be used. Defaults to None.

        Returns:
            he root node of the constructed decision tree.
        """

        if ds is None:
            ds = self.__ds
        
        # If no features left, return leaf node with most common output value
        if len(self.__get_ds_features(ds)) == 0:
            return Node(self.__get_most_common_output_value(ds))

        # Find most optimal feature
        feature = self.__find_optimal_feature(ds)
        tree = Node(feature)

        # Iterate over values of a feature
        for value in np.unique(ds[feature]):

            # Create subset with rows where feature equals to certain value
            subset = self.__get_subset(ds, feature, value)

            # Get output values and their quantity
            output_values, counts = np.unique(self.__get_output_column(ds), return_counts=True)

            # Check if quantity of one of output values exceeds threshold
            if (output_value := self.__get_output_value_threshold(output_values, counts)) is not None:

                # Add leaf with found output value
                tree.add_child(value, Node(output_value))
            else:

                # Call method recursively for a subset
                subtree = self.__build_tree(subset)

                # Add subtree
                tree.add_child(value, subtree)
        return tree

    def train(self):
        """
        Build the decision tree using the provided dataset.

        Returns:
            None
        """

        self.__tree = self.__build_tree()

    def predict(self, test_data):
        """
        Make predictions on new data using the trained decision tree.

        Args:
            test_data (dict): A dictionary containing feature values for prediction.

        Returns:
            str: The predicted class label.
        """

        def recursive_predict(node, data):
            if not node.children:
                return node.name
            feature_value = data[node.name]
            if feature_value in node.children:
                return recursive_predict(node.children[feature_value], data)
            return "Unknown"

        if self.__tree is None:
            raise ValueError("The decision tree has not been trained. Call the 'train' method first.")
        return recursive_predict(self.__tree, test_data)

    def score(self, test_ds: DataFrame):
        """
        Evaluate the accuracy of the decision tree on a test dataset.

        Args:
            test_ds (DataFrame): The test dataset containing features and true output labels.

        Returns:
            str: A string representing the accuracy score.
        """

        test_ds = list(test_ds.itertuples(index=False, name=None))
        test = [
            {
                key: val
                for key, val in zip(self.__get_ds_columns(self.__ds), vector)
            } for vector in test_ds
        ]

        total_count = 0
        correct_count = 0
        for row in test:
            total_count += 1
            output = row.pop(self.output_column_name)
            if self.predict(row) == output:
                correct_count += 1
        return f"Score: {correct_count * 100 / total_count}%"

    def __str__(self):
        return str(self.__tree)


if __name__ == "__main__":
    df = pd.read_csv("Hotel Reservations.csv").sample(frac=1, random_state=42).reset_index(drop=True)
    test_proportion = 0.2
    test_size = int(test_proportion * len(df))
    train_df, test_df = df[:-test_size], df[-test_size:]

    decision_tree = ID3DecisionTree(ds=train_df)
    decision_tree.train()

    print(decision_tree)
    print(decision_tree.score(test_df))
