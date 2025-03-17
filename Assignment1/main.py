import pandas as pd
from apyori import apriori

class Dataset:
    def __init__(self, csv_path):
        self.df = pd.read_csv(csv_path, parse_dates=["InvoiceDate"])
        self.stock_code_to_description = self.df.groupby("StockCode").agg({"Description": "first"}).to_dict()["Description"]
        self.csv_path = csv_path

    def get_grouped_data(self, group, data_item):
        orders_depending_on_invoice = self.df.groupby(group)[data_item].apply(lambda x: list(set(x))).tolist()
        return orders_depending_on_invoice
        orders_depending_on_Country = df.groupby("Invoice")["Country"].apply(list)
        orders_depending_on_Date = df.groupby("Invoice")["Country"].apply(list)

def print_association_results(association_results):
    results = []
    for item in association_results:
        # first index of the inner list
        # Contains base item and add item
        pair = item[0]
        items = [(x, dataset.stock_code_to_description[x]) for x in pair]
        # second index of the inner list
        value2 = str(item[1])[:7]

        # third index of the list located at 0th
        # of the third index of the inner list

        value3 = str(item[2][0][2])[:7]
        value4 = str(item[2][0][3])[:7]

        rows = (items, value2, value3, value4)
        results.append(rows)

    labels = ['Titles', 'Support', 'Confidence', 'Lift']
    movie_suggestion = pd.DataFrame.from_records(results, columns=labels)
    pd.set_option('display.max_colwidth', None)  # Show full content in each column
    pd.set_option('display.expand_frame_repr', False)  # Prevent line wrapping for large DataFrames
    pd.set_option('display.max_rows', None)  # Show all rows if needed
    print(movie_suggestion)

# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    dataset = Dataset("retail.csv")

    print("=== products often bought tougher ===")
    orders_depending_on_invoice = dataset.get_grouped_data("Invoice", "StockCode")
    association_results = list(apriori(orders_depending_on_invoice, min_support=0.01, min_confidence=0.5))
    print_association_results(association_results)

    print()
    print("=== Simular clients ===")
    orders_depending_on_invoice = dataset.get_grouped_data("Customer ID", "StockCode")
    association_results = list(apriori(orders_depending_on_invoice, min_support=0.09, min_confidence=0.7))
    print_association_results(association_results)

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
