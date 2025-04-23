from typing import MutableMapping

import pandas as pd
from sklearn.metrics import classification_report, roc_auc_score, accuracy_score, precision_score, recall_score
from sklearn.model_selection import train_test_split
from itertools import combinations
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm


class ClassifierTestFramework:
    def __init__(self, dataset, column_combinations):
        self.classifiers = {}
        self.classifiers_results = {}
        self.train_df, self.test_df = train_test_split(dataset, test_size=0.2, random_state=0)
        self.column_combinations = column_combinations
        pass

    def add_classifier(self, name, classifier):
        self.classifiers[name] = classifier
        self.train_and_test_classifier(name)

    def _all_subset(self, cols):
        for i in range(len(cols)):
            subset_items = list(combinations(set(cols), i + 1))
            for i, subset_list in enumerate(subset_items):
                yield list(subset_items[i]) + [i.replace("1st", "2nd") for i in subset_list if "1st" in i]

    def train_and_test_classifier(self, name):
        print("train and test for all combinations of features")
        self.classifiers_results[name] = {}
        for columns_name, columns in self.column_combinations.items():
            print(f"train and test for columns")

            X = self.train_df[columns].values.tolist()

            scaler = StandardScaler()
            X_scaled_array = scaler.fit_transform(X)

            # Wrap it back into a DataFrame
            X_scaled_df = pd.DataFrame(X_scaled_array, columns=self.train_df[columns].columns, index=self.train_df[columns].index)

            X = X_scaled_df

            Y = list(self.train_df["target"])

            cls = self.classifiers[name]
            cls.fit(X, Y)

            test_items = self.test_df[columns].values.tolist()

            scaler = StandardScaler()
            X_scaled_array = scaler.fit_transform(test_items)

            # Wrap it back into a DataFrame
            X_scaled_df = pd.DataFrame(X_scaled_array, columns=self.test_df[columns].columns,
                                       index=self.test_df[columns].index)

            test_items = X_scaled_df

            predictions = cls.predict(test_items)
            report = classification_report(self.test_df["target"], predictions)

            auc = roc_auc_score(self.test_df["target"], predictions)

            # 1. Accuracy
            accuracy = accuracy_score(self.test_df["target"], predictions)

            # 2. AUC
            auc = roc_auc_score(self.test_df["target"], predictions)

            # 3. Precision & Recall for both classes
            precision_0 = precision_score(self.test_df["target"], predictions, pos_label=0)
            recall_0 = recall_score(self.test_df["target"], predictions, pos_label=0)
            precision_1 = precision_score(self.test_df["target"], predictions, pos_label=1)
            recall_1 = recall_score(self.test_df["target"], predictions, pos_label=1)

            self.classifiers_results[name][f"{columns_name}"] = {"auc": auc, "accuracy": accuracy, "precision_0": precision_0, "recall_0": recall_0, "precision_1": precision_1, "recall_1": recall_1, "report": report}

    def get_results(self):
        return self.classifiers_results

    def get_results_by_name(self, name):
        return self.classifiers_results[name]

    def get_report(self):
        best_accuracy = (0.0, None)
        best_auc = (0.0, None)
        best_precision = (0.0, None)
        best_recall = (0.0, None)

        names = []
        all_results = []
        rows = []
        for key, items in self.classifiers_results.items():
            for columns, results in items.items():
                names.append(key + "_" + columns)
                all_results.append(results)
                rows.append([key + "_" + columns, results["auc"], results["accuracy"], results["precision_0"], results["recall_0"], results["precision_1"], results["recall_1"]])
                if results["auc"] > best_auc[0]:
                    best_auc = (results["auc"], (key, columns))
                if results["accuracy"] > best_precision[0]:
                    best_precision = (results["accuracy"], (key, columns))
                if results["precision_0"] > best_accuracy[0]:
                    best_accuracy = (results["precision_0"], (key, columns))
                if results["recall_0"] > best_recall[0]:
                    best_recall = (results["recall_0"], (key, columns))
        print("Best items")
        print("best_auc", best_auc)
        print("best_precision", best_precision)
        print("best_recall", best_recall)

        # Sort by AUC in descending order and get top 10

        df = pd.DataFrame(rows, columns=["name", "auc", "accuracy", "precision_0", "recall_0", "precision_1", "recall_1"])

        df.sort_values(["auc", "accuracy", "precision_0", "recall_0", "precision_1", "recall_1"], ascending=False, inplace=True)
        pd.set_option('display.max_columns', None)
        pd.set_option('display.max_rows', None)
        print(df)
        #
        # fig, axs = plt.subplots(1, 4, figsize=(10, 8), sharey=True)
        # axs[0].bar(top_names, [res["auc"] for res in top_results])
        # axs[1].bar(top_names, [res["accuracy"] for res in top_results])
        # axs[2].bar(top_names, [res["precision_0"] for res in top_results])
        # axs[3].bar(top_names, [res["recall_0"] for res in top_results])
        # plt.tight_layout()
        # plt.show()
