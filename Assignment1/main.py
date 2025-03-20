from datetime import datetime

import pandas as pd
from apyori import apriori

from dataset import Dataset

import pandas as pd

def print_association_rules(association_results):
    for item in association_results:
        pair = item.items  # The product combinations
        support = item.support  # Support value

        ordered_statistics = item.ordered_statistics[0]  # First rule (can iterate if needed)
        base_items = ordered_statistics.items_base  # LHS (if A, then B)
        add_items = ordered_statistics.items_add  # RHS (then B)
        confidence = ordered_statistics.confidence  # Confidence score
        lift = ordered_statistics.lift  # Lift metric
        print([x for x in base_items], " -> ", [x for x in add_items], " | ", f"confidence: {confidence:.2f}, support: {support:.2f}, lift: {lift:.2f}")


def print_association_results(association_results, dataset):
    results = []
    for item in association_results:
        pair = item.items  # The product combinations
        support = item.support  # Support value

        ordered_statistics = item.ordered_statistics[0]  # First rule (can iterate if needed)
        base_items = ordered_statistics.items_base  # LHS (if A, then B)
        add_items = ordered_statistics.items_add  # RHS (then B)
        confidence = ordered_statistics.confidence  # Confidence score
        lift = ordered_statistics.lift  # Lift metric

        # # Convert stock codes to readable product names if dataset is available
        # try:
        #     base_items = [dataset.stock_code_to_description[x] for x in base_items]
        #     add_items = [dataset.stock_code_to_description[x] for x in add_items]
        # except:
        base_items = list(base_items)
        add_items = list(add_items)

        results.append((pair, support, base_items, add_items, confidence, lift))

    # Create DataFrame
    df = pd.DataFrame(results, columns=['Items', 'Support', 'Base (If)', 'Add (Then)', 'Confidence', 'Lift'])

    # Display settings
    pd.set_option('display.max_colwidth', None)  # Show full content
    pd.set_option('display.expand_frame_repr', False)  # Prevent wrapping
    pd.set_option('display.max_rows', None)  # Show all rows

    print(df)


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    dataset = Dataset("retail.csv")
    dataset.handle_missing_or_wrong_values()
    dataset.categorize_data()
    #dataset.keep_last_x_items(100000)

    SUPPORT = 0.005
    CONFIDENCE = 0.6
    MIN_LIFT = 2

    features_configurations = [["quantityCat", "priceCat", "Customer ID"], ["quantityCat", "priceCat", "Country"],
                               ["Description", "quantityCat"], ["Description", "InvoiceDateCat"]]

    print(f"=== Test ===")
    dataset.df = dataset.df.astype(str)
    orders_depending_on_invoice = list(dataset.df[["Invoice","Description","quantityCat","InvoiceDateCat","priceCat","Customer ID"]].itertuples(index=False, name=None))
    print(orders_depending_on_invoice)
    print(f"Total order count: {len(orders_depending_on_invoice)}")
    print(f"Support: {SUPPORT} ({len(orders_depending_on_invoice) * SUPPORT} items needed)")
    print(f"Confidence: {len(orders_depending_on_invoice)}")
    print(f"Total order count: {len(orders_depending_on_invoice)}")
    association_results = list(
        apriori(orders_depending_on_invoice, min_support=SUPPORT, min_confidence=CONFIDENCE, min_lift=MIN_LIFT,
                max_length=5))
    print_association_rules(association_results)

    exit()
    pd.set_option('display.max_colwidth', None)  # Show full content in each column
    pd.set_option('display.expand_frame_repr', False)  # Prevent line wrapping for large DataFrames
    dataset = Dataset("retail.csv")
    dataset.handle_missing_or_wrong_values()
    dataset.categorize_data()
    dataset.keep_last_x_items(100000)
    print(dataset.df)

    print("=== products often bought tougher ===")
    orders_depending_on_invoice = dataset.get_grouped_data("Invoice", "StockCode")
    association_results = list(apriori(orders_depending_on_invoice, min_support=0.025, min_confidence=0.8, min_lift=2))
    print_association_results(association_results, dataset.df)


    exit(0)
    print("=== products bought at certain time moments ===")
    grouped_baskets = dataset.df.groupby(['InvoiceDateCat', 'StockCode']).size().reset_index(name='Count')

    print(grouped_baskets)
    print("Grouped data")
    association_results = list(apriori(grouped_baskets, min_support=0.02, min_confidence=0.5, min_lift=2))
    print_association_results(association_results, dataset.df)
    # exit(0)
    # print()
    # print("=== Simular clients ===")
    # orders_depending_on_invoice = dataset.get_grouped_data("Customer ID", "StockCode")orders_depending_on_invoice
    # association_results = list(apriori(orders_depending_on_invoice, min_support=0.09, min_confidence=0.7))
    # # Usage in your script
    # print_association_results(association_results, dataset.df)

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
