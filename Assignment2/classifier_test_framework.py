from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split


class ClassifierTestFramework:
    def __init__(self, dataset, columns):
        self.classifiers = {}
        self.classifiers_results = {}
        self.train_df, self.test_df = train_test_split(dataset, test_size=0.2)
        self.columns = columns
        pass

    def add_classifier(self, name, classifier):
        self.classifiers[name] = classifier
        self.train_and_test_classifier(name)

    def train_and_test_classifier(self, name):
        X = self.train_df[self.columns].values.tolist()
        Y = list(self.train_df["target"])

        cls = self.classifiers[name]
        cls.fit(X, Y)

        test_items = self.test_df[self.columns].values.tolist()
        predictions = cls.predict(test_items)
        report = classification_report(self.test_df["target"], predictions)
        self.classifiers_results[name] = report

    def get_results(self):
        return self.classifiers_results

    def get_results_by_name(self, name):
        return self.classifiers_results[name]